"""Prompt-conditional offset diagnostic for DAT.

For each test image we issue 5 prompts that target distinct spatial regions:
    UL = upper-left, UR = upper-right, LL = lower-left, LR = lower-right,
    NEUTRAL = generic description.
We capture conv_off_proj outputs (offsets in normalized [-1,1] coords) and
ask:

  Q1 (within-image-across-prompt vs cross-image-within-prompt variance):
      prompt_std   = average elementwise std of offsets across the 5 prompts,
                     for the same image
      image_std    = average elementwise std of offsets across the K images,
                     for the same prompt
      ratio = prompt_std / image_std
        ratio ~ 0  => prompts do not move offsets at all (prompt-blind)
        ratio ~ 1  => prompts move offsets as much as different images do
                      (strong language conditioning)

  Q2 (directional bias):
      For each prompt p, average offsets over (Lp*g, gh, gw) per image,
      then over images, giving (mean_dy_p, mean_dx_p).
      Convention: dy>0 means "look further DOWN", dx>0 means "look RIGHT".
      Expected for an ideally prompt-aware model:
        UL: dy<0, dx<0
        UR: dy<0, dx>0
        LL: dy>0, dx<0
        LR: dy>0, dx>0
        NEUTRAL: ~0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/root/autodl-tmp/lmms-eval")

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-no-eval")

from llava.model.language_model.modeling_qwen2_5vl_dat import (  # noqa: E402
    Qwen2_5_VLAttentionDAT,
)


PROMPTS = [
    ("UL",      "Describe in detail what you see in the upper-left region of the image."),
    ("UR",      "Describe in detail what you see in the upper-right region of the image."),
    ("LL",      "Describe in detail what you see in the lower-left region of the image."),
    ("LR",      "Describe in detail what you see in the lower-right region of the image."),
    ("NEUTRAL", "Describe what you see in the image."),
]
EXPECTED_SIGN = {
    "UL":      (-1, -1),
    "UR":      (-1, +1),
    "LL":      (+1, -1),
    "LR":      (+1, +1),
    "NEUTRAL": ( 0,  0),
}


# ============================================================================
# Hook
# ============================================================================
class OffsetCapture:
    def __init__(self):
        self.captures: dict[int, torch.Tensor] = {}

    def __call__(self, module, inputs, output):
        layer_idx = getattr(module, "_dat_diag_layer_idx", id(module))
        self.captures[layer_idx] = output.detach().float().cpu()


def install_hooks(model):
    cap = OffsetCapture()
    handles = []
    layer_indices = []
    for name, mod in model.named_modules():
        if isinstance(mod, Qwen2_5_VLAttentionDAT):
            try:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
            except Exception:
                layer_idx = id(mod)
            mod.conv_off_proj._dat_diag_layer_idx = layer_idx
            handles.append(mod.conv_off_proj.register_forward_hook(cap))
            layer_indices.append(layer_idx)
    return cap, handles, sorted(layer_indices)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/root/autodl-tmp/vldat_experiments/0515_full_1d5l_sa1b_ivcap_runB_spatial_warmup_rmsnorm-merged",
    )
    parser.add_argument("--n_images", type=int, default=4,
                        help="number of distinct HR-Bench 4K images to test")
    parser.add_argument("--out_dir", default="/root/autodl-tmp/diag_offsets_runB_prompt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] instantiating Qwen2_5_DATVL adapter from {args.ckpt}")
    from lmms_eval.models.simple.qwen2_5_dat_vl import Qwen2_5_DATVL
    adapter = Qwen2_5_DATVL(
        pretrained=args.ckpt,
        device=args.device,
        device_map=args.device,
        batch_size=1,
        attn_implementation="sdpa",
    )

    print("[2/4] installing offset capture hooks")
    capturer, handles, layer_indices = install_hooks(adapter._model)
    print(f"  hooks attached to layers: {layer_indices}")

    print("[3/4] loading HR-Bench 4K images")
    from datasets import load_dataset
    ds = load_dataset(
        "DreamMr/HR-Bench",
        "hrbench_version_split",
        split="hrbench_4k",
        cache_dir="/root/autodl-tmp/cache/huggingface/datasets",
    )
    # spread images across the dataset
    n_total = len(ds)
    chosen = [int(i * n_total / args.n_images) for i in range(args.n_images)]
    print(f"  selected image indices: {chosen}")

    from lmms_eval.api.instance import Instance
    from lmms_eval.tasks.hrbench.utils import (
        decode_base64_to_image, hrbench_doc_to_visual,
    )

    TASK = "hrbench4k"
    SPLIT = "hrbench_4k"

    class _DocLookup:
        def __init__(self, ds_obj):
            self._ds = ds_obj
        def __getitem__(self, idx):
            return self._ds[int(idx)]

    adapter.task_dict = {TASK: {SPLIT: _DocLookup(ds)}}

    # captures[(image_idx, prompt_id)][layer_idx] = np.array
    captures: dict[tuple[int, str], dict[int, np.ndarray]] = {}

    print(f"[4/4] running {len(chosen)} images x {len(PROMPTS)} prompts = "
          f"{len(chosen) * len(PROMPTS)} forward passes")

    for img_idx in chosen:
        doc = ds[img_idx]
        try:
            _ = decode_base64_to_image(doc["image"])  # sanity
        except Exception as e:
            print(f"  [WARN] image {img_idx} decode failed: {e}")
            continue

        for prompt_id, prompt_text in PROMPTS:
            instance = Instance(
                request_type="generate_until",
                arguments=(
                    prompt_text,
                    {"max_new_tokens": 1, "temperature": 0.0},
                    hrbench_doc_to_visual,
                    img_idx,
                    TASK,
                    SPLIT,
                ),
                idx=img_idx,
                metadata={"task": TASK, "doc_id": img_idx, "repeats": 0},
            )
            instance.doc = doc

            capturer.captures.clear()
            try:
                with torch.no_grad():
                    _ = adapter.generate_until([instance])
            except Exception as e:
                print(f"  [WARN] (img={img_idx}, p={prompt_id}) failed: "
                      f"{type(e).__name__}: {e}")
                continue

            if not capturer.captures:
                print(f"  [WARN] (img={img_idx}, p={prompt_id}) no capture")
                continue
            captures[(img_idx, prompt_id)] = {
                k: v.numpy() for k, v in capturer.captures.items()
            }
            print(f"  done (img={img_idx}, p={prompt_id})")

    if not captures:
        print("[ERR] no captures collected")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # Q1: within-image-across-prompt vs cross-image-within-prompt variance
    # ----------------------------------------------------------------------
    # For each image, stack the 5 prompts' offsets along axis 0; compute std
    # along that axis; mean over all elements -> prompt_std for that image.
    # Then average across images.
    prompt_std_per_layer = {L: [] for L in layer_indices}
    image_std_per_layer = {L: [] for L in layer_indices}

    for L in layer_indices:
        # within-image, across prompt
        for img_idx in chosen:
            stacks = []
            for prompt_id, _ in PROMPTS:
                if (img_idx, prompt_id) in captures and L in captures[(img_idx, prompt_id)]:
                    stacks.append(captures[(img_idx, prompt_id)][L])
            if len(stacks) < 2:
                continue
            try:
                shapes = {s.shape for s in stacks}
                if len(shapes) > 1:
                    # different Lp due to LR resize variability for the same image?
                    # For LR-first resize at fixed prompt-independent geometry,
                    # this should NOT happen. If it does, skip this layer/image.
                    continue
                arr = np.stack(stacks, axis=0)  # [P, Lp*g, 2, gh, gw]
                prompt_std_per_layer[L].append(float(arr.std(axis=0).mean()))
            except Exception:
                continue

        # cross-image, within prompt
        for prompt_id, _ in PROMPTS:
            stacks = []
            for img_idx in chosen:
                if (img_idx, prompt_id) in captures and L in captures[(img_idx, prompt_id)]:
                    stacks.append(captures[(img_idx, prompt_id)][L])
            if len(stacks) < 2:
                continue
            try:
                # Different images can have different Lp due to LR resize.
                # Compare elementwise only for the smallest common Lp slice.
                min_Lp = min(s.shape[0] for s in stacks)
                stacks_sliced = [s[:min_Lp] for s in stacks]
                shapes = {s.shape for s in stacks_sliced}
                if len(shapes) > 1:
                    continue
                arr = np.stack(stacks_sliced, axis=0)  # [I, min_Lp, 2, gh, gw]
                image_std_per_layer[L].append(float(arr.std(axis=0).mean()))
            except Exception:
                continue

    # ----------------------------------------------------------------------
    # Q2: directional bias per prompt
    # ----------------------------------------------------------------------
    # mean_dy / mean_dx per (prompt, layer), averaged over all elements over all images
    dir_bias_per_layer: dict[int, dict[str, tuple[float, float]]] = {L: {} for L in layer_indices}
    for L in layer_indices:
        for prompt_id, _ in PROMPTS:
            dys = []
            dxs = []
            for img_idx in chosen:
                if (img_idx, prompt_id) in captures and L in captures[(img_idx, prompt_id)]:
                    arr = captures[(img_idx, prompt_id)][L]  # [Lp*g, 2, gh, gw]
                    dys.append(float(arr[:, 0, :, :].mean()))
                    dxs.append(float(arr[:, 1, :, :].mean()))
            if dys and dxs:
                dir_bias_per_layer[L][prompt_id] = (
                    float(np.mean(dys)), float(np.mean(dxs))
                )

    # ----------------------------------------------------------------------
    # Print
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Q1: prompt_std vs image_std (lower = less responsive)")
    print("=" * 80)
    fmt = "{:>8s} | {:>14s} | {:>14s} | {:>10s}"
    print(fmt.format("layer", "prompt_std", "image_std", "ratio"))
    print("-" * 80)
    for L in layer_indices:
        ps = prompt_std_per_layer[L]
        is_ = image_std_per_layer[L]
        ps_avg = float(np.mean(ps)) if ps else float("nan")
        is_avg = float(np.mean(is_)) if is_ else float("nan")
        ratio = (ps_avg / is_avg) if (is_avg and is_avg > 0) else float("nan")
        print(fmt.format(f"L{L}", f"{ps_avg:.4f}", f"{is_avg:.4f}",
                         f"{ratio:.3f}" if np.isfinite(ratio) else "nan"))

    print("\nInterpretation:")
    print("  ratio ~ 0  => prompts do not move offsets (prompt-blind)")
    print("  ratio ~ 0.5 => prompts contribute moderately to offset variance")
    print("  ratio ~ 1  => prompts as influential as different images (strong)")

    print("\n" + "=" * 80)
    print("Q2: directional bias per (layer, prompt)  —  (mean_dy, mean_dx)")
    print("    expected:  UL=(-,-)  UR=(-,+)  LL=(+,-)  LR=(+,+)  NEUTRAL=(0,0)")
    print("=" * 80)
    header = ["layer"] + [pid for pid, _ in PROMPTS] + ["agree"]
    print("{:>6s} | ".format(header[0]) + " | ".join(
        f"{h:>14s}" for h in header[1:]
    ))
    print("-" * 110)
    for L in layer_indices:
        row = [f"L{L}"]
        agree = 0
        total = 0
        for pid, _ in PROMPTS:
            if pid in dir_bias_per_layer[L]:
                dy, dx = dir_bias_per_layer[L][pid]
                exp_dy, exp_dx = EXPECTED_SIGN[pid]
                if exp_dy != 0 and exp_dx != 0:
                    if (np.sign(dy) == exp_dy) and (np.sign(dx) == exp_dx):
                        agree += 1
                    total += 1
                row.append(f"({dy:+.3f},{dx:+.3f})")
            else:
                row.append("--")
        row.append(f"{agree}/{total}" if total else "--")
        print("{:>6s} | ".format(row[0]) + " | ".join(
            f"{x:>14s}" for x in row[1:]
        ))

    print()
    print("agree counts how many of the 4 directional prompts (UL/UR/LL/LR) gave")
    print("an offset whose sign in BOTH dy and dx matches the prompt direction.")
    print("  4/4  => offsets follow prompts perfectly")
    print("  >=2/4 => prompts have measurable directional effect")
    print("  <=1/4 => prompts have no consistent directional effect")

    # Persist
    out_json = {
        "ckpt": args.ckpt,
        "images": chosen,
        "prompts": [{"id": p[0], "text": p[1]} for p in PROMPTS],
        "Q1_prompt_std_per_layer": {
            f"L{L}": float(np.mean(prompt_std_per_layer[L])) if prompt_std_per_layer[L] else None
            for L in layer_indices
        },
        "Q1_image_std_per_layer": {
            f"L{L}": float(np.mean(image_std_per_layer[L])) if image_std_per_layer[L] else None
            for L in layer_indices
        },
        "Q2_directional_bias": {
            f"L{L}": {pid: list(dir_bias_per_layer[L].get(pid, (None, None)))
                     for pid, _ in PROMPTS}
            for L in layer_indices
        },
    }
    with open(out_dir / "offset_prompt_diag.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\njson dumped to {out_dir / 'offset_prompt_diag.json'}")

    for h in handles:
        h.remove()
    print("Done.")


if __name__ == "__main__":
    main()
