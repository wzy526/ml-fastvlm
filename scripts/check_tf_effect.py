#!/usr/bin/env python3
"""Preflight for teacher-forced HD sampling: does the forcing change what the
model sees? Run BEFORE launching an SFT that relies on `bbox` samples.

For a sample of bbox-carrying items it reproduces the trainer's LR-first
geometry (Qwen2VLCoupledDATDataset legacy mode) and the trainer's guards
(_teacher_force_window) and reports:
  - HD/LR pixel ratio          (1.0 = HD carries nothing LR lacks)
  - forced window area frac    (1.0 = window is the whole image = uniform grid)
  - fraction that would actually be forced under the guards
  - for synth data: text height at LR / HD in pixels

The 0915 tfbox run would have printed HD/LR ~1.0x and window ~1.0 here.

Usage:
  python scripts/check_tf_effect.py --json <train.json> --image_root ~/sft_data/train_split [--n 300]
"""
import argparse
import json
import math
import os
import random

from PIL import Image

FACTOR = 32


def lr_scale(W, H, lr_min, lr_max):
    px = W * H
    if px > lr_max:
        return math.sqrt(lr_max / px)
    if px < lr_min:
        return math.sqrt(lr_min / px)
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lr_min_pixels", type=int, default=200704)
    ap.add_argument("--lr_max_pixels", type=int, default=501760)
    ap.add_argument("--hd_max_pixels", type=int, default=5017600)
    ap.add_argument("--hr_scale", type=float, default=3.0)
    ap.add_argument("--min_cells", type=int, default=20)
    ap.add_argument("--min_hd_ratio", type=float, default=2.0)
    ap.add_argument("--max_window_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(args.json))
    boxed = [x for x in data if x.get("bbox")]
    print(f"{len(data)} samples, {len(boxed)} with bbox ({len(boxed) / max(1, len(data)):.1%})")
    if not boxed:
        return
    random.seed(args.seed)
    s = random.sample(boxed, min(args.n, len(boxed)))
    ratios, wins, forced, missing = [], [], 0, 0
    lr_txt, hd_txt = [], []
    for x in s:
        p = os.path.join(args.image_root, x["image"])
        if not os.path.exists(p):
            missing += 1
            if missing == 1:
                print("missing example:", p)
            continue
        W, H = Image.open(p).size
        s_lr = lr_scale(W, H, args.lr_min_pixels, args.lr_max_pixels)
        lr_px = (W * s_lr) * (H * s_lr)
        hd_px = min(lr_px * args.hr_scale ** 2, W * H, args.hd_max_pixels)
        s_hd = math.sqrt(hd_px / (W * H))
        ratio = hd_px / lr_px
        ratios.append(ratio)
        b = x["bbox"]; b = b[0] if isinstance(b[0], (list, tuple)) else b
        x0, y0, x1, y1 = [float(v) for v in b]
        if max(x0, y0, x1, y1) > 1.0:
            x0, x1, y0, y1 = x0 / W, x1 / W, y0 / H, y1 / H
        cells_w, cells_h = max(1, int(W * s_hd) // FACTOR), max(1, int(H * s_hd) // FACTOR)
        ww = min(1.0, max(x1 - x0, args.min_cells / cells_w))
        wh = min(1.0, max(y1 - y0, args.min_cells / cells_h))
        wins.append(ww * wh)
        forced += (ratio >= args.min_hd_ratio) and (ww * wh <= args.max_window_frac)
        if "text_height_px" in x:
            lr_txt.append(x["text_height_px"] * s_lr); hd_txt.append(x["text_height_px"] * s_hd)

    def q(v, f):
        v = sorted(v); return v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")
    n = len(ratios)
    print(f"checked {n} (missing {missing})")
    print(f"HD/LR pixel ratio : median {q(ratios, .5):.2f}x  p10 {q(ratios, .1):.2f}x  "
          f"share >= {args.min_hd_ratio}x: {sum(r >= args.min_hd_ratio for r in ratios) / n:.0%}")
    print(f"window area frac  : median {q(wins, .5):.2f}   p90 {q(wins, .9):.2f}   "
          f"share <= {args.max_window_frac}: {sum(w <= args.max_window_frac for w in wins) / n:.0%}")
    print(f"WOULD BE FORCED   : {forced / n:.0%} of bbox samples "
          f"({forced / n * len(boxed) / max(1, len(data)):.1%} of the whole dataset)")
    if lr_txt:
        print(f"text height       : LR median {q(lr_txt, .5):.1f} px   HD median {q(hd_txt, .5):.1f} px")
    if forced / n < 0.8:
        print("\n!! most bbox samples would NOT be forced -- fix the data before training")


if __name__ == "__main__":
    main()
