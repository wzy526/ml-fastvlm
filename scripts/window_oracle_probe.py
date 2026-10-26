#!/usr/bin/env python3
"""Ceiling for LEARNED window-level offsets (centre shift + scale per window).

Per sample, the head's live window w (side sw x sh) is the base. Two candidate
sets, each 11 windows including w, are scored by the answer loss (CE over the
option letters at the first answer token, GT letter) on the live single-pass
prompt [LR, q, hint, crop(window), q]:

  struct  w shifted by {-0.5, 0, +0.5} x its side in x and y (9), plus w
          scaled x0.7 / x1.4 about its centre: what a window-offset head
          could choose
  random  10 windows of w's size at uniform random positions, plus w: the
          control. Picking the lowest-GT-loss crop out of several inflates
          accuracy on its own (2-4 options); only struct - random is the
          headroom that structured offsets add.

Reported: base acc, best-by-loss acc and any-correct rate for both sets,
which struct candidate wins, mean loss of base vs best.

    for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i setsid nohup python scripts/window_oracle_probe.py \
        --model_path $E/checkpoint-1000 --where_head_pt $E/loc/where_lm8_ce20.pt --dataset vstar --shard $i --shards 8 \
        --out $E/loc/wop/vstar_$i.json > $E/loc/wop/vstar_$i.log 2>&1 < /dev/null & disown; done
    python scripts/window_oracle_probe.py --summarize "$E/loc/wop/vstar_*.json"
"""
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SHIFTS = [(dx, dy) for dy in (-0.5, 0.0, 0.5) for dx in (-0.5, 0.0, 0.5)]
SCALES = [0.7, 1.4]
STRUCT_NAMES = [f"shift{dx:+.1f},{dy:+.1f}" for dx, dy in SHIFTS] + [f"scale{s}" for s in SCALES]


def place(cx, cy, w, h):
    w, h = min(w, 1.0), min(h, 1.0)
    x0 = min(max(cx - w / 2, 0.0), 1.0 - w)
    y0 = min(max(cy - h / 2, 0.0), 1.0 - h)
    return (x0, y0, x0 + w, y0 + h)


def struct_windows(win):
    w, h = win[2] - win[0], win[3] - win[1]
    cx, cy = (win[0] + win[2]) / 2, (win[1] + win[3]) / 2
    out = [tuple(win) if (dx, dy) == (0.0, 0.0) else place(cx + dx * w, cy + dy * h, w, h)
           for dx, dy in SHIFTS]
    out += [place(cx, cy, w * s, h * s) for s in SCALES]
    return out                                   # index 4 = shift(0,0) = win itself


def random_windows(win, n, rng):
    w, h = win[2] - win[0], win[3] - win[1]
    return [tuple(win)] + [place(rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2), w, h)
                           for _ in range(n)]


def summarize(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        rows.extend(json.load(open(f)))
    if not rows:
        raise SystemExit(f"no json matched {pattern}")
    n = len(rows)

    def report(name, rs):
        m = len(rs)
        if not m:
            return
        base = sum(r["struct"][4]["ok"] for r in rs)
        sb = sum(min(r["struct"], key=lambda c: c["loss"])["ok"] for r in rs)
        rb = sum(min(r["random"], key=lambda c: c["loss"])["ok"] for r in rs)
        sa = sum(any(c["ok"] for c in r["struct"]) for r in rs)
        ra = sum(any(c["ok"] for c in r["random"]) for r in rs)
        bl = sum(r["struct"][4]["loss"] for r in rs) / m
        sl = sum(min(c["loss"] for c in r["struct"]) for r in rs) / m
        rl = sum(min(c["loss"] for c in r["random"]) for r in rs) / m
        print(f"  {name:22s} n={m:4d}  base {base:4d}  best-by-loss struct {sb:4d} random {rb:4d} (headroom {sb - rb:+d})  "
              f"any-correct struct {sa:4d} random {ra:4d}  loss base {bl:.3f} best struct {sl:.3f} random {rl:.3f}")
    print(f"{n} samples")
    report("all", rows)
    for c in sorted({r["cat"] for r in rows}):
        report(c, [r for r in rows if r["cat"] == c])
    report("base wrong", [r for r in rows if not r["struct"][4]["ok"]])
    wins = {}
    for r in rows:
        k = STRUCT_NAMES[min(range(len(r["struct"])), key=lambda j: r["struct"][j]["loss"])]
        wins[k] = wins.get(k, 0) + 1
    print("struct winner (lowest loss): " + "  ".join(f"{k} {v}" for k, v in sorted(wins.items(), key=lambda kv: -kv[1])))
    wins = {}
    for r in rows:
        if r["struct"][4]["ok"]:
            continue
        for j, c in enumerate(r["struct"]):
            if c["ok"]:
                wins[STRUCT_NAMES[j]] = wins.get(STRUCT_NAMES[j], 0) + 1
    print("base wrong -> candidate right: " + "  ".join(f"{k} {v}" for k, v in sorted(wins.items(), key=lambda kv: -kv[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--model_path")
    ap.add_argument("--where_head_pt", default=None)
    ap.add_argument("--dataset", choices=["vstar", "hrbench4k", "hrbench8k"], default="vstar")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--hrbench_parquet", default=None)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--crop_tok", type=int, default=400)
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--n_random", type=int, default=10)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from probe_whd_qwen35 import load_samples
    import llava.model.language_model.modeling_qwen3_5_dat as dat_mod

    dev = "cuda"
    model = dat_mod.Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=args.attn).eval()
    if args.where_head_pt:
        sd = torch.load(os.path.expanduser(args.where_head_pt), map_location="cpu")
        sd = {k[len("where_head."):] if k.startswith("where_head.") else k: v for k, v in sd.items()}
        model.where_head.load_state_dict(sd)
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=args.lr_tok * 1024)
    hr_processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=1024, max_pixels=100_000_000)
    tok = hr_processor.tokenizer

    samples = load_samples(argparse.Namespace(dataset=args.dataset, vstar_root=args.vstar_root,
                                              hrbench_parquet=args.hrbench_parquet, image_folder=None,
                                              max_samples=args.max_samples))
    mine = [(i, s) for i, s in enumerate(samples) if i % args.shards == args.shard]
    print(f"[wop] {args.dataset} shard {args.shard}/{args.shards}: {len(mine)} samples", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)

    @torch.inference_mode()
    def score(image, lr_img, prompt, win, letter_ids, gt_idx):
        crop = dat_mod._where_crop_image(image, win, args.crop_tok)
        q = {"type": "text", "text": prompt}
        content = [{"type": "image", "image": lr_img}, q, {"type": "text", "text": dat_mod.WHERE_HINT},
                   {"type": "image", "image": crop}, dict(q)]
        msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
        text = hr_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = hr_processor(text=[text], images=[lr_img, crop], return_tensors="pt")
        ids = inp["input_ids"].to(dev)
        thw = inp["image_grid_thw"].to(dev)
        mm = torch.zeros_like(ids)
        mm[ids == model.config.image_token_id] = 1
        pos3d, rope_deltas = model.model.get_rope_index(ids, mm, image_grid_thw=thw)
        model.model.rope_deltas = rope_deltas
        o = model(input_ids=ids, position_ids=pos3d, mm_token_type_ids=mm,
                  pixel_values=inp["pixel_values"].to(dev, torch.bfloat16), image_grid_thw=thw,
                  use_cache=False, return_dict=True)
        lg = o.logits[0, -1, letter_ids].float()
        return {"loss": float(-torch.log_softmax(lg, -1)[gt_idx]), "ok": int(lg.argmax()) == gt_idx}

    rows = []
    for gi, s in mine:
        image, prompt = s["image"], s["prompt"]
        rec = model.where_crop_generate(image, prompt, processor, hr_processor, where_question=prompt,
                                        crop_tok=args.crop_tok, head_prefix="train", single_pass=True,
                                        max_new_tokens=1)
        win = tuple(rec["window"])
        thw_lr = processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0]
        lr_img = image.resize((int(thw_lr[2]) * 16, int(thw_lr[1]) * 16), Image.BICUBIC)
        letters = re.findall(r"^\(?([A-Z])[.)]", prompt, flags=re.M) or ["A", "B", "C", "D"]
        letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in letters]
        if s["gt"] not in letters:
            print(f"[wop] {gi}: gt {s['gt']} not in options {letters}; skip", flush=True)
            continue
        gt_idx = letters.index(s["gt"])
        rng = random.Random(gi)
        sw = struct_windows(win)
        st = [dict(score(image, lr_img, prompt, w, letter_ids, gt_idx), win=list(w)) for w in sw]
        rw = random_windows(win, args.n_random, rng)
        rd = [st[4]] + [dict(score(image, lr_img, prompt, w, letter_ids, gt_idx), win=list(w)) for w in rw[1:]]
        rows.append({"i": gi, "cat": s["category"], "gt": s["gt"], "prompt": prompt,
                     "window": list(win), "struct": st, "random": rd})
        best = min(range(len(st)), key=lambda j: st[j]["loss"])
        print(f"[wop] {gi} {s['category'][:10]} base ok={int(st[4]['ok'])} loss={st[4]['loss']:.3f}  "
              f"struct best {STRUCT_NAMES[best]} ok={int(st[best]['ok'])} loss={st[best]['loss']:.3f}  "
              f"random best ok={int(min(rd, key=lambda c: c['loss'])['ok'])} loss={min(c['loss'] for c in rd):.3f}",
              flush=True)
        if args.out and len(rows) % 20 == 0:
            json.dump(rows, open(os.path.expanduser(args.out), "w"))

    if args.out:
        json.dump(rows, open(os.path.expanduser(args.out), "w"))
        print(f"[wop] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
