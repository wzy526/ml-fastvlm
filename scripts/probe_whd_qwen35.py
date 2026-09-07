#!/usr/bin/env python3
"""HD-pathway diagnostic for Qwen3.5-DAT checkpoints.

Answers two questions about a trained ckpt, mechanistically (no benchmark run):

  1. How much attention mass actually goes to HD keys?
     Hooks ``_merge_two_pass_lse`` and records the per-layer LSE merge weight
     w_hd = exp(lse_hd - logaddexp(lse_lr, lse_hd)) over answer positions.
     If w_hd is ~0 everywhere, the HD pathway is inert regardless of what the
     HD features contain.

  2. Does removing HD change the model's *behavior*?
     Greedy-generates each sample twice (with / without pixel_values_hd) and
     reports how many answers differ. If w_hd is small AND answers are
     identical, HD-off benchmark parity is a foregone conclusion.

Geometry (LR-first) is identical to training / the lmms-eval wrapper.

Usage (single GPU, on the eval cluster):
  CUDA_VISIBLE_DEVICES=0 python scripts/probe_whd_qwen35.py \
      --model_path ~/vldat_experiments/0826_sft_qwen35_2b_dat_genvs-merged \
      --dataset hrbench4k --max_samples 50 --tok_budget 256 \
      --out whd_hr4k_tok256.json
"""

import argparse
import base64
import io
import json
import math
import os
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE          # 32
TOK_PX = FACTOR * FACTOR        # 1024

SYSTEM_PROMPT = "You are a helpful assistant."


def load_samples(args):
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
    from datasets import load_dataset
    split = {"hrbench4k": "hrbench_4k", "hrbench8k": "hrbench_8k"}[args.dataset]
    ds = load_dataset("DreamMr/HR-Bench", "hrbench_version_split", split=split)
    for doc in ds:
        if len(samples) >= args.max_samples:
            break
        img = Image.open(io.BytesIO(base64.b64decode(doc["image"]))).convert("RGB")
        q = doc["question"].strip() + "\nAnswer the option letter directly."
        samples.append({"image": img, "question": q})
    return samples


def hd_target_size(image, lr_grid_thw, hr_scale, hd_cap):
    lr_px = int(lr_grid_thw[1]) * FACTOR * int(lr_grid_thw[2]) * FACTOR
    hd_total = min(lr_px * hr_scale * hr_scale, image.width * image.height, hd_cap)
    aspect = image.width / image.height
    hd_h = int(math.sqrt(hd_total / aspect))
    hd_w = int(hd_h * aspect)
    hd_h = max(FACTOR, (hd_h // FACTOR) * FACTOR)
    hd_w = max(FACTOR, (hd_w // FACTOR) * FACTOR)
    return hd_w, hd_h


def build_inputs(sample, processor, hr_processor, args, device, dtype):
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

    hd_w, hd_h = hd_target_size(
        sample["image"], inputs["image_grid_thw"][0], args.hr_scale, args.hd_cap)
    hd_img = sample["image"].resize((hd_w, hd_h), Image.BICUBIC)
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


# ── w_hd recorder ─────────────────────────────────────────────
WHD = defaultdict(list)   # layer_idx -> list of (mean, p90, max) per merge call
RECORD = {"on": False}


def install_whd_hook():
    from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5AttentionDAT
    import torch.nn.functional as F

    orig = Qwen3_5AttentionDAT._merge_two_pass_lse

    def patched(self, out1, lse1, out2, lse2, ans_start, ans_end):
        if RECORD["on"]:
            with torch.no_grad():
                l1 = lse1[:, :, ans_start:ans_end].float()
                l2 = lse2.float()
                if self.hd_gate is not None:
                    l2 = l2 + F.logsigmoid(self.hd_gate)
                w_hd = (l2 - torch.logaddexp(l1, l2)).exp().flatten()
                WHD[self.layer_idx].append((
                    w_hd.mean().item(),
                    w_hd.quantile(0.9).item(),
                    w_hd.max().item(),
                ))
        return orig(self, out1, lse1, out2, lse2, ans_start, ans_end)

    Qwen3_5AttentionDAT._merge_two_pass_lse = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--processor_path", default=None)
    ap.add_argument("--dataset", choices=["hrbench4k", "hrbench8k"], default="hrbench4k")
    ap.add_argument("--image_folder", default=None)
    ap.add_argument("--tok_budget", type=int, default=256,
                    help="LR LLM-visual-token budget (lr_max_pixels = tok*1024). "
                         "Use a LOW budget: that's where HD should matter most.")
    ap.add_argument("--min_pixels", type=int, default=28224)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--hr_scale", type=int, default=3)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen3_5_dat import (
        Qwen3_5DATForConditionalGeneration,
    )

    install_whd_hook()

    samples = load_samples(args)
    print(f"[probe] {len(samples)} samples, tok_budget={args.tok_budget}")

    model = Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn,
    ).eval()
    ppath = args.processor_path or args.model_path
    processor = AutoProcessor.from_pretrained(
        ppath, min_pixels=args.min_pixels, max_pixels=args.tok_budget * TOK_PX)
    hr_processor = AutoProcessor.from_pretrained(
        ppath, min_pixels=TOK_PX, max_pixels=100_000_000)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                      temperature=None, top_p=None, top_k=None)

    n_diff = 0
    pairs = []
    for s in tqdm(samples, desc="probe"):
        base_inputs, hd_extra = build_inputs(s, processor, hr_processor, args, device, dtype)

        # Pass 1: HD ON, record w_hd
        RECORD["on"] = True
        with torch.inference_mode():
            out_on = model.generate(**base_inputs, **hd_extra, **gen_kwargs)
        RECORD["on"] = False

        # Pass 2: HD OFF (no pixel_values_hd at all)
        with torch.inference_mode():
            out_off = model.generate(**base_inputs, **gen_kwargs)

        n_in = base_inputs["input_ids"].shape[1]
        txt_on = processor.tokenizer.decode(out_on[0][n_in:], skip_special_tokens=True).strip()
        txt_off = processor.tokenizer.decode(out_off[0][n_in:], skip_special_tokens=True).strip()
        if txt_on != txt_off:
            n_diff += 1
        pairs.append({"q": s["question"][:80], "hd_on": txt_on, "hd_off": txt_off})

    # ── report ────────────────────────────────────────────────
    print(f"\n==== w_hd per DAT layer (over {len(samples)} samples, "
          f"tok_budget={args.tok_budget}) ====")
    print(f"{'layer':>6} | {'mean':>8} | {'p90':>8} | {'max':>8} | {'#calls':>7}")
    print("-" * 50)
    layer_stats = {}
    for lid in sorted(WHD):
        ms = [x[0] for x in WHD[lid]]
        p9 = [x[1] for x in WHD[lid]]
        mx = [x[2] for x in WHD[lid]]
        layer_stats[lid] = {
            "mean": sum(ms) / len(ms),
            "p90": sum(p9) / len(p9),
            "max": max(mx),
            "calls": len(ms),
        }
        st = layer_stats[lid]
        print(f"{lid:>6} | {st['mean']:>8.4f} | {st['p90']:>8.4f} | "
              f"{st['max']:>8.4f} | {st['calls']:>7}")

    frac = n_diff / max(len(samples), 1)
    print(f"\nanswers changed by HD-off: {n_diff}/{len(samples)} ({frac:.0%})")
    print("interpretation:")
    print("  w_hd mean ~0 + answers identical  -> HD pathway inert (self-locked)")
    print("  w_hd sizable + answers identical  -> HD attended but redundant with LR")
    print("  answers differ a lot              -> HD pathway IS load-bearing; check")
    print("                                       whether the diffs are right/wrong")

    if args.out:
        json.dump({
            "model_path": args.model_path,
            "dataset": args.image_folder or args.dataset,
            "tok_budget": args.tok_budget,
            "num_samples": len(samples),
            "answers_changed": n_diff,
            "layer_whd": layer_stats,
            "pairs": pairs,
        }, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"[probe] saved -> {args.out}")


if __name__ == "__main__":
    main()
