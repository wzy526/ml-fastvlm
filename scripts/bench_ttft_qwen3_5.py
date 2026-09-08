#!/usr/bin/env python3
"""TTFT benchmark: Qwen3.5 base vs Qwen3.5-DAT, on real HRBench images.

Ports the measurement protocol of test_qwen2vl_ttft_flops.py (warmup, then
per-sample cuda.synchronize-bracketed PREFILL forward, use_cache=False) to the
Qwen3.5 family, adding:

  - dual mode: --model_type base | dat  (DAT = two-pass LSE ckpt from ml-fastvlm)
  - a sweep over LLM-visual-token budgets (--tokens, 1 token = 32*32 = 1024 px)
  - DAT LR-first geometry identical to training / the lmms-eval wrapper:
        lr band = [min_pixels, tok*1024]
        hd_total = min(lr_pixels * hr_scale^2, orig_pixels, hd_cap),
        aspect-recovered and floor-snapped to factor 32
  - peak VRAM per point.

Only PREFILL is timed: all DAT overhead (HD ViT forward + HD cross-attn) lives
there; decode steps never touch HD KV (not cached), so decode speed is
identical to base by construction.

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_ttft_qwen3_5.py \
      --model_type base --model_path /path/to/Qwen3.5-2B \
      --dataset hrbench4k --max_samples 100 --out ttft_base_hr4k.json

  CUDA_VISIBLE_DEVICES=0 python scripts/bench_ttft_qwen3_5.py \
      --model_type dat --model_path /path/to/0826_sft_qwen35_2b_dat_genvs-merged \
      --dataset hrbench4k --max_samples 100 --out ttft_dat_hr4k.json
"""

import argparse
import base64
import io
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE          # 32
TOK_PX = FACTOR * FACTOR        # 1024 px per merged LLM visual token

SYSTEM_PROMPT = "You are a helpful assistant."


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────

def load_samples(args):
    """Return list of {image: PIL, question: str}."""
    samples = []
    if args.image_folder:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = sorted(f for f in os.listdir(args.image_folder)
                       if os.path.splitext(f)[1].lower() in exts)[: args.max_samples]
        for f in files:
            samples.append({
                "image": Image.open(os.path.join(args.image_folder, f)).convert("RGB"),
                "question": "Describe this image briefly.",
            })
        return samples

    # HRBench via HF datasets (cached on the eval cluster from prior sweeps).
    from datasets import load_dataset
    split = {"hrbench4k": "hrbench_4k", "hrbench8k": "hrbench_8k"}[args.dataset]
    ds = load_dataset("DreamMr/HR-Bench", "hrbench_version_split", split=split)
    seen_imgs = 0
    for doc in ds:
        if seen_imgs >= args.max_samples:
            break
        img = Image.open(io.BytesIO(base64.b64decode(doc["image"]))).convert("RGB")
        q = doc["question"].strip() + "\nAnswer the option letter directly."
        samples.append({"image": img, "question": q})
        seen_imgs += 1
    return samples


# ──────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────

def load_model(args):
    common = dict(torch_dtype=torch.bfloat16, device_map={"": 0},
                  attn_implementation=args.attn)
    if args.model_type == "base":
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_path, **common)
    else:
        from llava.model.language_model.modeling_qwen3_5_dat import (
            Qwen3_5DATForConditionalGeneration,
        )
        model = Qwen3_5DATForConditionalGeneration.from_pretrained(args.model_path, **common)
    return model.eval()


def make_processor(path, min_px, max_px):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(path, min_pixels=min_px, max_pixels=max_px)


# ──────────────────────────────────────────────────────────────
# DAT LR-first HR geometry (mirrors training & the lmms-eval wrapper)
# ──────────────────────────────────────────────────────────────

def hd_target_size(image, lr_grid_thw, hr_scale, hd_cap):
    """Return (hd_w, hd_h): HR resize target, factor-32 snapped."""
    lr_px = int(lr_grid_thw[1]) * FACTOR * int(lr_grid_thw[2]) * FACTOR
    hd_total = lr_px * hr_scale * hr_scale
    hd_total = min(hd_total, image.width * image.height, hd_cap)
    aspect = image.width / image.height
    hd_h = int(math.sqrt(hd_total / aspect))
    hd_w = int(hd_h * aspect)
    hd_h = max(FACTOR, (hd_h // FACTOR) * FACTOR)
    hd_w = max(FACTOR, (hd_w // FACTOR) * FACTOR)
    return hd_w, hd_h


# ──────────────────────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────────────────────

def build_inputs(sample, processor, hr_processor, args, device, dtype):
    """Tokenize + preprocess one sample. NOT timed (mirrors old script)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": sample["question"]},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = processor(text=[text], images=[sample["image"]], return_tensors="pt")

    extra = {}
    if args.model_type == "dat":
        hd_w, hd_h = hd_target_size(
            sample["image"], inputs["image_grid_thw"][0], args.hr_scale, args.hd_cap)
        hd_img = sample["image"].resize((hd_w, hd_h), Image.BICUBIC)
        hr_inputs = hr_processor.image_processor(images=[hd_img], return_tensors="pt")
        extra["pixel_values_hd"] = hr_inputs["pixel_values"].to(device=device, dtype=dtype)
        extra["image_grid_thw_hd"] = hr_inputs["image_grid_thw"].to(device)

    moved = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k == "pixel_values":
            moved[k] = v.to(device=device, dtype=dtype)
        elif k in ("input_ids", "attention_mask", "image_grid_thw"):
            moved[k] = v.to(device)
    moved.update(extra)
    return moved


def measure_prefill(model, inputs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(**inputs, use_cache=False)
        _ = out.logits[:, -1, :].argmax(dim=-1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def run_point(model, samples, args, tok_budget, processor_path):
    """One token-budget point: rebuild processors, loop samples, aggregate."""
    lr_max = tok_budget * TOK_PX
    processor = make_processor(processor_path, args.min_pixels, lr_max)
    hr_processor = None
    if args.model_type == "dat":
        # Wide band: HR size is forced per-image by exact resize upstream.
        hr_processor = make_processor(processor_path, TOK_PX, 100_000_000)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Warmup (first fla/flash kernels compile here, not in the timed loop)
    for s in samples[: args.warmup]:
        inputs = build_inputs(s, processor, hr_processor, args, device, dtype)
        measure_prefill(model, inputs)

    ttft, lr_toks, hd_toks = [], [], []
    for s in tqdm(samples, desc=f"tok{tok_budget}", leave=False):
        inputs = build_inputs(s, processor, hr_processor, args, device, dtype)
        thw = inputs["image_grid_thw"][0]
        lr_toks.append(int(thw[1]) // MERGE * (int(thw[2]) // MERGE))
        if "image_grid_thw_hd" in inputs:
            thw_hd = inputs["image_grid_thw_hd"][0]
            hd_toks.append(int(thw_hd[1]) // MERGE * (int(thw_hd[2]) // MERGE))
        ttft.append(measure_prefill(model, inputs))

    return {
        "tok_budget": tok_budget,
        "lr_max_pixels": lr_max,
        "num_samples": len(ttft),
        "avg_ttft_ms": float(np.mean(ttft)),
        "median_ttft_ms": float(np.median(ttft)),
        "p95_ttft_ms": float(np.percentile(ttft, 95)),
        "avg_llm_visual_tokens": float(np.mean(lr_toks)),
        "avg_hd_tokens": float(np.mean(hd_toks)) if hd_toks else 0.0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["base", "dat"], required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--processor_path", default=None,
                    help="defaults to model_path (merged DAT ckpts ship a processor)")
    ap.add_argument("--dataset", choices=["hrbench4k", "hrbench8k"], default="hrbench4k")
    ap.add_argument("--image_folder", default=None,
                    help="bypass HF dataset; scan a local image dir instead")
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[256, 640, 1280, 2560, 6400, 11520])
    ap.add_argument("--min_pixels", type=int, default=32768)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--hr_scale", type=int, default=3)
    ap.add_argument("--attn", default="flash_attention_2",
                    help="LR-path attention impl for BOTH models (DAT's two-pass "
                         "LSE helpers always use flash internally)")
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    samples = load_samples(args)
    print(f"[bench] {len(samples)} samples from "
          f"{args.image_folder or args.dataset}; model_type={args.model_type}")
    model = load_model(args)
    processor_path = args.processor_path or args.model_path

    rows = []
    for tok in args.tokens:
        rows.append(run_point(model, samples, args, tok, processor_path))
        r = rows[-1]
        print(f"  tok{tok:>6}: avg {r['avg_ttft_ms']:8.1f} ms  "
              f"p50 {r['median_ttft_ms']:8.1f}  p95 {r['p95_ttft_ms']:8.1f}  "
              f"lr_tok {r['avg_llm_visual_tokens']:7.0f}  hd_tok {r['avg_hd_tokens']:7.0f}  "
              f"vram {r['peak_vram_gb']:.1f}G")

    result = {
        "model_type": args.model_type,
        "model_path": args.model_path,
        "dataset": args.image_folder or args.dataset,
        "attn": args.attn,
        "hd_cap": args.hd_cap if args.model_type == "dat" else None,
        "hr_scale": args.hr_scale if args.model_type == "dat" else None,
        "gpu": torch.cuda.get_device_name(0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "points": rows,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[bench] saved -> {args.out}")


if __name__ == "__main__":
    main()
