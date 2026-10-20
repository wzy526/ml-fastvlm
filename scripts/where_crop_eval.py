#!/usr/bin/env python3
"""Score model.where_crop_generate on V*. The head is read from the checkpoint
(where_head.pt next to config.json). The live path this has to match scored
V* 77.5 at LR 640 (the external-pt run; off is 64.4).

  cp -n $E/loc/where_lm8_mix.pt $E/checkpoint-1000/where_head.pt
  CUDA_VISIBLE_DEVICES=0 python scripts/where_crop_eval.py \
      --model_path $E/checkpoint-1000 \
      --shard 0 --shards 8 --out $E/loc/where_crop2_0.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_whd_qwen35 import extract_letter, load_vstar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--out", default=None)
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen3_5_dat import (
        Qwen3_5DATForConditionalGeneration,
    )

    model = Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn).eval()
    if not model._where_head_loaded:
        raise SystemExit(f"no where-head in {args.model_path} (expected where_head.pt)")
    c = model.config.text_config.hidden_size
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=1024, max_pixels=args.lr_tok * 1024)
    hr_processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=1024, max_pixels=100_000_000)

    probe_args = argparse.Namespace(vstar_root=args.vstar_root, max_samples=args.max_samples)
    samples = load_vstar(probe_args)
    mine = [s for i, s in enumerate(samples) if i % args.shards == args.shard]
    print(f"[where_crop] shard {args.shard}/{args.shards}: {len(mine)}/{len(samples)} samples, "
          f"hidden {c}, layer {args.layer}, min_cells {args.min_cells}", flush=True)

    rows, n_ok = [], 0
    per = {}
    for i, s in enumerate(mine):
        where_q = s["prompt"].split("\n")[0]
        rec = model.where_crop_generate(
            s["image"], s["prompt"], processor, hr_processor,
            where_question=where_q, layer=args.layer, min_cells=args.min_cells)
        pred = extract_letter(rec["text"])
        ok = pred == s["gt"]
        n_ok += ok
        cat = s["category"]
        slot = per.setdefault(cat, [0, 0])
        slot[0] += 1
        slot[1] += ok
        win = rec["window"]
        rows.append({"i": args.shard + i * args.shards, "pred": pred, "gt": s["gt"],
                     "ok": bool(ok), "cat": cat,
                     "box": list(rec["box"]), "window": list(win),
                     "area": (win[2] - win[0]) * (win[3] - win[1])})
        print(f"[where_crop] {rows[-1]['i']} pred={pred} gt={s['gt']} ok={int(ok)} "
              f"cat={cat} area={rows[-1]['area']:.3f}", flush=True)
    n = len(rows)
    print(f"[where_crop] shard {args.shard} acc {n_ok}/{n} ({100 * n_ok / max(n, 1):.1f}%)", flush=True)
    for cat, (m, hit) in sorted(per.items()):
        print(f"[where_crop]   {cat}: {hit}/{m}", flush=True)
    if args.out:
        json.dump(rows, open(os.path.expanduser(args.out), "w"))
        print(f"[where_crop] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
