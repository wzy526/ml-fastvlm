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

from probe_whd_qwen35 import extract_letter, load_samples


class OracleHead(torch.nn.Module):
    """Stands in for model.where_head: a one-hot map at the GT union-box centre.
    Everything downstream (square, pad, crop, prompt) is unchanged, so this is
    the ceiling of THIS pipeline given a perfect head, per question.
    An nn.Module because where_head is a registered child of the model."""

    def __init__(self):
        super().__init__()
        self.centre = None

    def set_sample(self, s):
        W, H = s["image"].size
        bb = s["bboxes"]
        x0, y0 = min(b[0] for b in bb), min(b[1] for b in bb)
        x1, y1 = max(b[0] + b[2] for b in bb), max(b[1] + b[3] for b in bb)
        self.centre = ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H)

    def forward(self, feat, txt):
        gh, gw, _ = feat.shape
        logits = torch.full((gh, gw), -1e4, device=feat.device)
        cx, cy = self.centre
        logits[min(int(cy * gh), gh - 1), min(int(cx * gw), gw - 1)] = 0.0
        return logits


@torch.inference_mode()
def plain_generate(model, processor, image, prompt, max_new_tokens=8):
    """where_crop_generate's answer call without the hint/crop: [system, image, question]."""
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    inp = processor(text=[text], images=[image], return_tensors="pt")
    kw = {k: inp[k].to(model.device) for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
    gen = model.generate(**kw, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None)
    n = kw["input_ids"].shape[1]
    return {"text": processor.tokenizer.decode(gen[0][n:], skip_special_tokens=True).strip(),
            "box": (0.0, 0.0, 1.0, 1.0), "window": (0.0, 0.0, 1.0, 1.0), "where_grid": ()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset", choices=["vstar", "hrbench4k", "hrbench8k"], default="vstar")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--hrbench_parquet", default=None, help="see probe_whd_qwen35.py")
    ap.add_argument("--max_samples", type=int, default=300,
                    help="V* has 191; HR-Bench 4K/8K 800 each -> raise this")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--ground_pad", type=float, default=2.0)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--where_tok", type=int, default=0,
                    help="LR token budget of the head's prefill; 0 = same as --lr_tok")
    ap.add_argument("--out", default=None)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--oracle_box", action="store_true",
                    help="replace the head with the GT box centre (pipeline ceiling)")
    ap.add_argument("--no_crop", action="store_true",
                    help="baseline: same system prompt / LR image / question, no head, no crop (= off@lr_tok)")
    ap.add_argument("--where_head_pt", default=None,
                    help="override the head weights with this XAttnHead state_dict "
                         "(e.g. the 640 head while <ckpt>/where_head.pt holds the 1600 one)")
    ap.add_argument("--head_prefix", default="legacy", choices=["legacy", "train"],
                    help="text span the head reads: legacy = question + generation prompt "
                         "(scripts/where_head.py), train = question + hint (train_qwen_dat.py crops)")
    args = ap.parse_args()

    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen3_5_dat import (
        Qwen3_5DATForConditionalGeneration,
    )

    model = Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn).eval()
    if not model._where_head_loaded and not args.no_crop:
        raise SystemExit(f"no where-head in {args.model_path} (expected where_head.pt)")
    if args.where_head_pt:
        sd = torch.load(args.where_head_pt, map_location="cpu")
        sd = {k[len("where_head."):] if k.startswith("where_head.") else k: v for k, v in sd.items()}
        model.where_head.load_state_dict(sd)
        print(f"[where_crop] head weights <- {args.where_head_pt}", flush=True)
    oracle = None
    if args.oracle_box:
        oracle = OracleHead()
        model.where_head = oracle
    c = model.config.text_config.hidden_size
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=1024, max_pixels=args.lr_tok * 1024)
    hr_processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=1024, max_pixels=100_000_000)
    where_processor = None
    if args.where_tok and args.where_tok != args.lr_tok:
        where_processor = AutoProcessor.from_pretrained(
            args.model_path, min_pixels=1024, max_pixels=args.where_tok * 1024)

    probe_args = argparse.Namespace(dataset=args.dataset, vstar_root=args.vstar_root,
                                    hrbench_parquet=args.hrbench_parquet, image_folder=None,
                                    max_samples=args.max_samples)
    samples = load_samples(probe_args)
    mine = [(i, s) for i, s in enumerate(samples) if i % args.shards == args.shard]
    if oracle is not None:
        if args.dataset != "vstar":
            raise SystemExit("--oracle_box needs GT boxes; only V* has them")
        mine = [(i, s) for i, s in mine if s["bboxes"]]
    print(f"[where_crop] shard {args.shard}/{args.shards}: {len(mine)}/{len(samples)} samples, "
          f"{'ORACLE box centre, ' if oracle is not None else ''}"
          f"hidden {c}, layer {args.layer}, min_cells {args.min_cells}, "
          f"ground_pad {args.ground_pad}, where_tok {args.where_tok or args.lr_tok}, "
          f"head_prefix {args.head_prefix}", flush=True)

    rows, n_ok = [], 0
    per = {}
    for gi, s in mine:
        where_q = s["prompt"].split("\n")[0]
        if oracle is not None:
            oracle.set_sample(s)
        if args.no_crop:
            rec = plain_generate(model, processor, s["image"], s["prompt"])
        else:
            rec = model.where_crop_generate(
                s["image"], s["prompt"], processor, hr_processor,
                where_question=where_q, layer=args.layer, min_cells=args.min_cells,
                ground_pad=args.ground_pad, where_processor=where_processor,
                head_prefix=args.head_prefix)
        pred = extract_letter(rec["text"])
        ok = pred == s["gt"]
        n_ok += ok
        cat = s["category"]
        slot = per.setdefault(cat, [0, 0])
        slot[0] += 1
        slot[1] += ok
        win = rec["window"]
        rows.append({"i": gi, "pred": pred, "gt": s["gt"],
                     "ok": bool(ok), "cat": cat,
                     "box": list(rec["box"]), "window": list(win),
                     "where_grid": list(rec.get("where_grid", ())),
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
