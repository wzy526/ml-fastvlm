#!/usr/bin/env python3
"""Quick sanity test of the 0526 pretrain ckpt.

What this tests
---------------
1. Caption quality on N held-out SA-1B images, side-by-side vs base
   Qwen2.5-VL-3B-Instruct. Tests whether the projector's 0.08 loss
   reduction translates into visibly different caption style/content.

2. sample_locs visualization per DAT layer at inference time.
   Confirms the offset-collapse pathology we diagnosed: deep DAT
   layers (L18, L24, L30) should still be pinning sample_locs to
   the image border because this ckpt was trained with the buggy
   Kaiming init for conv_off_proj. After the init fix is applied
   and the model is re-pretrained, the same script should show
   sample_locs scattered inside the image.

What this does NOT test
-----------------------
- Q&A benchmarks (LLM frozen → essentially same as base, no signal)
- HD-detail tasks (DAT is noisy in this ckpt → results misleading)

Usage
-----
    python scripts/qwen2_5vl_adl_0526/_test_pretrain_caption.py \
        --ckpt /root/autodl-tmp/vldat_experiments/0526_pretrain_sa1b_caption_lse_ste \
        --num_images 5

Output: ``<ckpt>/_test_outputs/captions.md`` and ``<ckpt>/_test_outputs/sampling_*.png``.
"""
import argparse
import json
import os
import random
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure we import the project's DAT model class, not the stock HF one.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from llava.model.language_model.modeling_qwen2_5vl_dat import (
    Qwen2_5_VLAttentionDAT,
    Qwen2_5_VLDATForConditionalGeneration,
)


CAPTION_PROMPT = (
    "Generate an accurate, single-paragraph description based on the given "
    "image. Do not omit any visible objects, attributes, spatial relations, "
    "or text. Be specific about colors, materials, and scene context."
)


def load_dat_ckpt(ckpt_path, device, dtype):
    """Load the pretrain ckpt with the DAT model class.

    The ckpt's config.json has architectures=["Qwen2_5_VLDATForConditionalGeneration"]
    so AutoModelForVision2Seq would fail (class not registered in HF). Use
    the project class directly. DAT params are in the ckpt's safetensors.
    """
    print(f"[load] Loading DAT ckpt: {ckpt_path}")
    model = Qwen2_5_VLDATForConditionalGeneration.from_pretrained(
        ckpt_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    print(f"[load]   {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    return model


def load_base(base_path, device, dtype):
    print(f"[load] Loading base: {base_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    return model


def install_sample_locs_hook(dat_model):
    """Register a hook on every DAT attention module to capture
    ``sample_locs`` from ``_generate_offsets_and_sample`` on each forward.

    Stores the latest sample_locs tensor on the module as ``_last_sample_locs``
    and the layer index (decoder layer order) as ``_dat_layer_idx``.
    """
    dat_layers = []
    for name, module in dat_model.named_modules():
        if isinstance(module, Qwen2_5_VLAttentionDAT):
            # Layer idx parsed from "model.language_model.layers.<idx>.self_attn"
            idx = int(name.split('.layers.')[-1].split('.')[0])
            module._dat_layer_idx = idx
            module._last_sample_locs = None

            orig_fn = module._generate_offsets_and_sample

            def make_wrapped(orig, mod):
                def wrapped(*args, **kwargs):
                    key_hd, value_hd, sampling_locs_out = orig(*args, **kwargs)
                    # sampling_locs_out: [Lp, off_grps, gh, gw, 2]
                    mod._last_sample_locs = sampling_locs_out.detach().cpu()
                    return key_hd, value_hd, sampling_locs_out
                return wrapped

            module._generate_offsets_and_sample = make_wrapped(orig_fn, module)
            dat_layers.append((idx, module))
    dat_layers.sort(key=lambda x: x[0])
    print(f"[hook] Installed sample_locs capture on {len(dat_layers)} DAT layers: "
          f"{[idx for idx, _ in dat_layers]}")
    return dat_layers


def build_messages(image_path, prompt_text):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ],
    }]


def caption_one(model, processor, image_path, device, dtype, max_new_tokens=200):
    messages = build_messages(image_path, CAPTION_PROMPT)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
    ).to(device)
    if dtype == torch.bfloat16:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                inputs[k] = v.to(dtype)

    # The Qwen2.5-VL processor emits ``mm_token_type_ids`` for distinguishing
    # image vs text tokens. The stock Qwen2_5_VLForConditionalGeneration
    # accepts it, but our Qwen2_5_VLDATForConditionalGeneration.forward does
    # not (HF's generate validates kwargs strictly and raises ValueError).
    # Drop it — DAT inference uses input_ids + attention_mask + pixel_values
    # + image_grid_thw which together fully specify the multimodal input.
    inputs.pop("mm_token_type_ids", None)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    new_ids = out_ids[0, inputs.input_ids.shape[1]:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def plot_sample_locs(image_path, layer_idx, sample_locs, out_path):
    """Plot sample_locs (in [-1, 1] normalized coords) on top of the
    original image. Each off_grp gets a distinct hue.

    sample_locs shape: [Lp, off_grps, gh, gw, 2] (we plot lp=0).
    """
    img = np.asarray(Image.open(image_path).convert("RGB"))
    H_pix, W_pix = img.shape[:2]
    locs = sample_locs[0].numpy()  # [off_grps, gh, gw, 2]
    n_grps = locs.shape[0]

    MAX_FIG_IN = 8.0
    ar = W_pix / max(H_pix, 1)
    if ar >= 1:
        fig_w, fig_h = MAX_FIG_IN, MAX_FIG_IN / ar
    else:
        fig_h, fig_w = MAX_FIG_IN, MAX_FIG_IN * ar

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(img, extent=[0, W_pix, H_pix, 0], aspect='equal', alpha=0.85)
    for g in range(n_grps):
        hue = g / max(n_grps, 1)
        x_norm = locs[g, :, :, 0].flatten()
        y_norm = locs[g, :, :, 1].flatten()
        x_pix = (x_norm + 1.0) * 0.5 * W_pix
        y_pix = (y_norm + 1.0) * 0.5 * H_pix
        color = mcolors.hsv_to_rgb([hue, 1.0, 1.0])
        ax.scatter(
            x_pix, y_pix, c=[color], s=40,
            edgecolors='white', linewidths=0.5, zorder=3,
        )
    ax.set_xlim(-0.02 * W_pix, 1.02 * W_pix)
    ax.set_ylim(1.02 * H_pix, -0.02 * H_pix)
    ax.set_aspect('equal')
    ax.set_title(f"L{layer_idx}  sample_locs (off_grps × grid_size²)", fontsize=10)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def sample_images(sa1b_root, n, seed=42):
    """Pick N random images from the latest shards (least likely to be
    'memorized' even by a 1-epoch run, since they were seen latest)."""
    shards = sorted(
        d for d in os.listdir(sa1b_root) if d.startswith("sa_")
    )[-5:]  # take last 5 shards
    rng = random.Random(seed)
    picks = []
    for _ in range(n):
        shard = rng.choice(shards)
        jpgs = os.listdir(os.path.join(sa1b_root, shard))
        jpgs = [f for f in jpgs if f.endswith(".jpg")]
        fname = rng.choice(jpgs)
        picks.append(os.path.join(sa1b_root, shard, fname))
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to pretrain ckpt dir.")
    ap.add_argument("--base", default="/root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct",
                    help="Base Qwen2.5-VL-3B-Instruct (no DAT) for comparison.")
    ap.add_argument("--sa1b_root", default="/root/autodl-tmp/models_data/sa1b_images")
    ap.add_argument("--num_images", type=int, default=5)
    ap.add_argument("--out_dir", default=None,
                    help="Default: <repo>/_test_outputs/<basename(ckpt)>/. "
                         "Kept inside the repo (not in the ckpt dir) so test "
                         "artifacts don't bloat the checkpoint directory and "
                         "are easy to gitignore / diff across runs.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        ckpt_name = os.path.basename(os.path.normpath(args.ckpt))
        out_dir = REPO_ROOT / "_test_outputs" / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] device={args.device}  dtype={args.dtype}  out_dir={out_dir}")

    processor = AutoProcessor.from_pretrained(args.base)

    images = sample_images(args.sa1b_root, args.num_images)
    print(f"[setup] Selected {len(images)} test images:")
    for p in images:
        print(f"  {p}")

    # ---- 1. Caption with base ----
    base = load_base(args.base, args.device, dtype)
    base_caps = []
    for i, p in enumerate(images):
        cap = caption_one(base, processor, p, args.device, dtype)
        print(f"\n[base   img {i}] {os.path.basename(p)}\n  {cap[:200]}")
        base_caps.append(cap)
    del base
    torch.cuda.empty_cache()

    # ---- 2. Caption with pretrain ckpt + capture sample_locs on first img ----
    dat = load_dat_ckpt(args.ckpt, args.device, dtype)
    dat_layers = install_sample_locs_hook(dat)

    dat_caps = []
    captured_locs = None
    for i, p in enumerate(images):
        cap = caption_one(dat, processor, p, args.device, dtype)
        print(f"\n[pretrain img {i}] {os.path.basename(p)}\n  {cap[:200]}")
        dat_caps.append(cap)
        if i == 0:
            captured_locs = {
                idx: mod._last_sample_locs
                for idx, mod in dat_layers
                if mod._last_sample_locs is not None
            }

    # ---- 3. Save text side-by-side ----
    # Copy images into out_dir/images/ so the markdown can reference them
    # with relative paths — absolute filesystem paths (/root/...) do not
    # render in most markdown viewers (Cursor preview, GitHub, etc.).
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    local_imgs = []
    for p in images:
        dst = img_dir / os.path.basename(p)
        if not dst.exists():
            shutil.copy2(p, dst)
        local_imgs.append(dst)

    md_path = out_dir / "captions.md"
    with open(md_path, "w") as f:
        f.write(f"# Pretrain ckpt vs base — caption side-by-side\n\n")
        f.write(f"- ckpt: `{args.ckpt}`\n- base: `{args.base}`\n\n")
        for i, (p, lp, bc, dc) in enumerate(zip(images, local_imgs, base_caps, dat_caps)):
            rel = os.path.relpath(lp, out_dir)
            f.write(f"## Image {i}: `{os.path.basename(p)}`\n\n")
            f.write(f"![]({rel})\n\n")
            f.write(f"### Base\n\n{textwrap.fill(bc, 100)}\n\n")
            f.write(f"### Pretrain\n\n{textwrap.fill(dc, 100)}\n\n---\n\n")
    print(f"\n[save] {md_path}")

    # ---- 4. Plot sample_locs for first image, per DAT layer ----
    if captured_locs is not None:
        first_img = images[0]
        for idx in sorted(captured_locs.keys()):
            locs = captured_locs[idx]
            png_path = out_dir / f"sampling_L{idx:02d}.png"
            plot_sample_locs(first_img, idx, locs, png_path)
            print(f"[save] {png_path}")
    else:
        print("[warn] No sample_locs captured. DAT hook may not have fired "
              "(check whether the model actually used the DAT path).")

    print(f"\n[done] All outputs in {out_dir}")


if __name__ == "__main__":
    main()
