#!/usr/bin/env python3
"""HD-pathway diagnostics for Qwen3.5-DAT checkpoints (inference-only).

Runs each HR-Bench sample under several HD configurations and reports
accuracy (overall + per category FSP/FCP), answer flips vs HD-off, and the
per-layer LSE merge weight w_hd = exp(lse_hd - logaddexp(lse_lr, lse_hd)).

Configs (all share the same LR input; only the HD side changes):
  off               no pixel_values_hd at all (vanilla single-pass attention)
  <source> b=<bias> HD pass on, with
      --hd_source  real     the real image at HD resolution (default)
                   lr_up    LR-resized image upsampled back to HD size
                            (same key count, ZERO extra information)
                   shuffle  a different sample's image (wrong content)
                   noise    uniform random noise
      --hd_bias    constant(s) added to lse_hd before the merge; >0 pushes
                   attention toward HD, <0 away. Pass several values to
                   sweep, e.g. --hd_bias -6 -3 0 1.5 3

Diagnosis A (bias sweep, real source): does any b>0 help? -> HD content is
  useful but mis-weighted (gate fix suffices). Does only b<0 help? -> HD
  content is noise (feature-side problem).
Diagnosis B (source ablation, b=0): lr_up/shuffle/noise ~= real -> the model
  treats HD as a content-free attention sink, w_hd is scale-driven.

Prompt / option formatting / GT match lmms-eval's hrbench task exactly.

Usage (single GPU, on the eval cluster):
  export HF_ENDPOINT=https://hf-mirror.com
  CUDA_VISIBLE_DEVICES=0 python scripts/probe_whd_qwen35.py \
      --model_path /data/oss_bucket_0/wangziyi/vldat_experiments/0826_sft_qwen35_2b_dat_genvs-merged \
      --dataset hrbench4k --max_samples 200 --tok_budget 256 \
      --hd_source real --hd_bias -6 -3 0 1.5 3 --out diagA_hr4k_tok256.json
"""

import argparse
import base64
import glob
import io
import json
import math
import os
import re
import string
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE          # 32
TOK_PX = FACTOR * FACTOR        # 1024

SYSTEM_PROMPT = "You are a helpful assistant."
LETTER_RE = re.compile(r"\b([A-H])\b")


# ──────────────────────────────────────────────────────────────
# Data (mirrors lmms_eval/tasks/hrbench/utils.py)
# ──────────────────────────────────────────────────────────────

def _is_nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def load_samples(args):
    samples = []
    if args.image_folder:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = sorted(f for f in os.listdir(args.image_folder)
                       if os.path.splitext(f)[1].lower() in exts)[: args.max_samples]
        for f in files:
            samples.append({
                "image": Image.open(os.path.join(args.image_folder, f)).convert("RGB"),
                "prompt": "Describe this image briefly.",
                "gt": None, "category": "n/a",
            })
        return samples

    from datasets import load_dataset
    split = {"hrbench4k": "hrbench_4k", "hrbench8k": "hrbench_8k"}[args.dataset]
    # Prefer the parquet already sitting in the HF hub cache (or an explicit
    # --hrbench_parquet): load_dataset("DreamMr/HR-Bench") always phones the
    # Hub to resolve the repo first, which hangs for HF_HUB_DOWNLOAD_TIMEOUT
    # when the mirror is slow and fails outright under HF_HUB_OFFLINE=1 once
    # the ~/.cache/huggingface/datasets arrow cache has been wiped.
    pq = args.hrbench_parquet or _local_hrbench_parquet(split)
    if pq:
        print(f"[probe] HR-Bench from local parquet: {pq}")
        ds = load_dataset("parquet", data_files=pq, split="train")
    else:
        print("[probe] HR-Bench parquet not in local hub cache; resolving via the Hub")
        ds = load_dataset("DreamMr/HR-Bench", "hrbench_version_split", split=split)
    for doc in ds:
        if len(samples) >= args.max_samples:
            break
        img = Image.open(io.BytesIO(base64.b64decode(doc["image"]))).convert("RGB")
        options = {c: doc[c] for c in string.ascii_uppercase
                   if c in doc and not _is_nan(doc[c])}
        opt_txt = "".join(f"{k}. {v}\n" for k, v in options.items())
        prompt = f"{doc['question'].strip()}\n{opt_txt}Answer the option letter directly."
        samples.append({
            "image": img, "prompt": prompt,
            "gt": str(doc["answer"]).strip().upper(),
            "category": str(doc.get("category", "n/a")),
        })
    return samples


def _local_hrbench_parquet(split):
    """Locate hr_bench_{4k,8k}.parquet inside the HF hub cache, if downloaded."""
    fname = {"hrbench_4k": "hr_bench_4k.parquet", "hrbench_8k": "hr_bench_8k.parquet"}[split]
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    hits = sorted(glob.glob(os.path.join(hub, "datasets--DreamMr--HR-Bench", "snapshots", "*", fname)),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def extract_letter(text):
    m = LETTER_RE.search(text.strip().upper())
    return m.group(1) if m else "Z"


# ──────────────────────────────────────────────────────────────
# Geometry (LR-first, identical to training / lmms-eval wrapper)
# ──────────────────────────────────────────────────────────────

def hd_target_size(image, lr_grid_thw, hr_scale, hd_cap):
    lr_px = int(lr_grid_thw[1]) * FACTOR * int(lr_grid_thw[2]) * FACTOR
    hd_total = min(lr_px * hr_scale * hr_scale, image.width * image.height, hd_cap)
    aspect = image.width / image.height
    hd_h = int(math.sqrt(hd_total / aspect))
    hd_w = int(hd_h * aspect)
    hd_h = max(FACTOR, (hd_h // FACTOR) * FACTOR)
    hd_w = max(FACTOR, (hd_w // FACTOR) * FACTOR)
    return hd_w, hd_h


def make_hd_image(sample, idx, samples, lr_thw, hd_w, hd_h, source):
    """Build the HD-side image under the requested ablation source."""
    img = sample["image"]
    if source == "real":
        return img.resize((hd_w, hd_h), Image.BICUBIC)
    if source == "lr_up":
        # exactly what the LR branch sees (patch grid * 16 px), blown back up
        lr_w_px = int(lr_thw[2]) * PATCH
        lr_h_px = int(lr_thw[1]) * PATCH
        return (img.resize((lr_w_px, lr_h_px), Image.BICUBIC)
                   .resize((hd_w, hd_h), Image.BICUBIC))
    if source == "shuffle":
        other = samples[(idx + 1) % len(samples)]["image"]
        return other.resize((hd_w, hd_h), Image.BICUBIC)
    if source == "noise":
        rng = np.random.RandomState(idx)
        arr = rng.randint(0, 256, size=(hd_h, hd_w, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")
    raise ValueError(source)


def build_inputs(sample, idx, samples, processor, hr_processor, args, device, dtype):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": sample["prompt"]},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=[text], images=[sample["image"]], return_tensors="pt")
    lr_thw = inputs["image_grid_thw"][0]

    hd_w, hd_h = hd_target_size(sample["image"], lr_thw, args.hr_scale, args.hd_cap)
    hd_img = make_hd_image(sample, idx, samples, lr_thw, hd_w, hd_h, args.hd_source)
    hr_inputs = hr_processor.image_processor(images=[hd_img], return_tensors="pt")

    moved = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k == "pixel_values":
            moved[k] = v.to(device=device, dtype=dtype)
        elif k in ("input_ids", "attention_mask", "image_grid_thw"):
            moved[k] = v.to(device)
    hd_extra = {
        "pixel_values_hd": hr_inputs["pixel_values"].to(device=device, dtype=dtype),
        "image_grid_thw_hd": hr_inputs["image_grid_thw"].to(device),
    }
    return moved, hd_extra


# ──────────────────────────────────────────────────────────────
# Merge hook: w_hd recorder + lse_hd bias injector
# ──────────────────────────────────────────────────────────────
WHD = defaultdict(list)          # layer_idx -> [(mean, p90, max), ...]
STATE = {"record": False, "bias": 0.0}


def install_merge_hook():
    from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5AttentionDAT
    import torch.nn.functional as F

    orig = Qwen3_5AttentionDAT._merge_two_pass_lse

    def patched(self, out1, lse1, out2, lse2, ans_start, ans_end):
        if STATE["bias"] != 0.0:
            lse2 = lse2 + STATE["bias"]
        if STATE["record"]:
            with torch.no_grad():
                l1 = lse1[:, :, ans_start:ans_end].float()
                l2 = lse2.float()
                if self.hd_gate is not None:
                    l2 = l2 + F.logsigmoid(self.hd_gate)
                w = (l2 - torch.logaddexp(l1, l2)).exp().flatten()
                WHD[self.layer_idx].append(
                    (w.mean().item(), w.quantile(0.9).item(), w.max().item()))
        return orig(self, out1, lse1, out2, lse2, ans_start, ans_end)

    Qwen3_5AttentionDAT._merge_two_pass_lse = patched


# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--processor_path", default=None)
    ap.add_argument("--dataset", choices=["hrbench4k", "hrbench8k"], default="hrbench4k")
    ap.add_argument("--image_folder", default=None)
    ap.add_argument("--hrbench_parquet", default=None,
                    help="explicit path to hr_bench_4k/8k.parquet (skips the Hub entirely)")
    ap.add_argument("--tok_budget", type=int, default=256,
                    help="LR LLM-visual-token budget (lr_max_pixels = tok*1024)")
    ap.add_argument("--min_pixels", type=int, default=28224)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--hr_scale", type=int, default=3)
    ap.add_argument("--hd_source", choices=["real", "lr_up", "shuffle", "noise"],
                    default="real")
    ap.add_argument("--hd_bias", type=float, nargs="+", default=[0.0],
                    help="constant(s) added to lse_hd; several values = sweep")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen3_5_dat import (
        Qwen3_5DATForConditionalGeneration,
    )

    install_merge_hook()
    samples = load_samples(args)
    has_gt = samples and samples[0]["gt"] is not None
    print(f"[probe] {len(samples)} samples  tok_budget={args.tok_budget}  "
          f"hd_source={args.hd_source}  biases={args.hd_bias}")

    model = Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn,
    ).eval()
    ppath = args.processor_path or args.model_path
    processor = AutoProcessor.from_pretrained(
        ppath, min_pixels=args.min_pixels, max_pixels=args.tok_budget * TOK_PX)
    hr_processor = AutoProcessor.from_pretrained(ppath, min_pixels=TOK_PX,
                                                 max_pixels=100_000_000)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                      temperature=None, top_p=None, top_k=None)

    # config name -> list of predicted letters (parallel to samples)
    configs = ["off"] + [f"{args.hd_source} b={b:+g}" for b in args.hd_bias]
    preds = {c: [] for c in configs}
    raw = {c: [] for c in configs}
    record_bias = 0.0 if 0.0 in args.hd_bias else args.hd_bias[0]

    def gen(inputs):
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        n_in = inputs["input_ids"].shape[1]
        return processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()

    for i, s in enumerate(tqdm(samples, desc="probe")):
        base_inputs, hd_extra = build_inputs(
            s, i, samples, processor, hr_processor, args, device, dtype)

        STATE["bias"] = 0.0
        t = gen(base_inputs)
        raw["off"].append(t); preds["off"].append(extract_letter(t))

        for b, cname in zip(args.hd_bias, configs[1:]):
            STATE["bias"] = float(b)
            STATE["record"] = (b == record_bias)
            t = gen({**base_inputs, **hd_extra})
            STATE["record"] = False
            raw[cname].append(t); preds[cname].append(extract_letter(t))
    STATE["bias"] = 0.0

    # ── report ────────────────────────────────────────────────
    cats = sorted({s["category"] for s in samples})
    gts = [s["gt"] for s in samples]
    n = len(samples)

    def acc(letters, mask=None):
        idx = [k for k in range(n) if mask is None or mask[k]]
        if not idx:
            return float("nan")
        return 100.0 * sum(letters[k] == gts[k] for k in idx) / len(idx)

    print(f"\n==== accuracy on {n} samples  (tok_budget={args.tok_budget}, "
          f"hd_source={args.hd_source}) ====")
    hdr = f"{'config':>16} | {'acc':>6} | " + " ".join(f"{c[:8]:>8}" for c in cats) \
        + f" | {'flip_vs_off':>11} | {'unres':>5}"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for c in configs:
        row = {"acc": acc(preds[c]) if has_gt else None}
        for cat in cats:
            mask = [s["category"] == cat for s in samples]
            row[f"acc_{cat}"] = acc(preds[c], mask) if has_gt else None
        row["flip_vs_off"] = sum(p != q for p, q in zip(preds[c], preds["off"]))
        row["unresolved"] = sum(p == "Z" for p in preds[c])
        summary[c] = row
        cat_cells = " ".join(
            f"{row[f'acc_{cat}']:>8.2f}" if has_gt else f"{'—':>8}" for cat in cats)
        acc_cell = f"{row['acc']:>6.2f}" if has_gt else f"{'—':>6}"
        print(f"{c:>16} | {acc_cell} | {cat_cells} | {row['flip_vs_off']:>11} | "
              f"{row['unresolved']:>5}")

    if has_gt and len(configs) > 1:
        # right->wrong / wrong->right decomposition for each HD config vs off
        print("\n==== flips vs off (HD-on made it ...) ====")
        for c in configs[1:]:
            fixed = sum(preds[c][k] == gts[k] != preds["off"][k] for k in range(n))
            broke = sum(preds["off"][k] == gts[k] != preds[c][k] for k in range(n))
            summary[c]["fixed"] = fixed; summary[c]["broke"] = broke
            print(f"{c:>16} : fixed {fixed:>3}   broke {broke:>3}   net {fixed - broke:+d}")

    print(f"\n==== w_hd per DAT layer  (source={args.hd_source}, bias={record_bias:+g}) ====")
    print(f"{'layer':>6} | {'mean':>8} | {'p90':>8} | {'max':>8}")
    print("-" * 40)
    layer_whd = {}
    for lid in sorted(WHD):
        ms, p9, mx = zip(*WHD[lid])
        layer_whd[lid] = {"mean": sum(ms) / len(ms), "p90": sum(p9) / len(p9), "max": max(mx)}
        st = layer_whd[lid]
        print(f"{lid:>6} | {st['mean']:>8.4f} | {st['p90']:>8.4f} | {st['max']:>8.4f}")

    if args.out:
        json.dump({
            "model_path": args.model_path,
            "dataset": args.image_folder or args.dataset,
            "tok_budget": args.tok_budget,
            "hd_source": args.hd_source,
            "hd_bias": args.hd_bias,
            "num_samples": n,
            "summary": summary,
            "layer_whd": layer_whd,
            "per_sample": [
                {"gt": gts[k], "category": samples[k]["category"],
                 **{c: raw[c][k] for c in configs}}
                for k in range(n)
            ],
        }, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"[probe] saved -> {args.out}")


if __name__ == "__main__":
    main()
