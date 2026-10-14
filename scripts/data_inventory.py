#!/usr/bin/env python3
"""Inventory of the bbox-carrying SFT data on the cluster, from the point of
view of a two-stage localiser (question -> coarse region on the LR image,
crop + question -> precise box): how many samples carry a box, how large the
images are, and how big the target is at the LR resolution.

For every LLaVA-format json found (top-level ``bbox`` field = [x0, y0, x1, y1]
image fractions, or boxes written in the conversation text), prints:
  n / n_bbox          samples, samples with a usable box
  img long side       median native image long side (px), share >= 1500 px
  box side (frac)     median / p10 of sqrt(box area) as a fraction of the image
  box @LR640 (px)     median / p10 of the target's side in pixels when the image
                      is shown at a 640-token LR (what the stage-1 localiser sees;
                      < ~16 px = invisible, the V* regime)
  box @stage2 (px)    median target side inside a 4x-box / 20-cell crop at 640
                      tokens (what the stage-2 localiser sees)
  V*-like share       samples whose target is < 16 px at LR640
Image sizes are read from the headers of up to --sample images per json.

Usage (cluster):
  export OSS_DATA=/data/oss_bucket_0/wangziyi/models_data
  python scripts/data_inventory.py --root $OSS_DATA --image_root ~/sft_data/train_split
  python scripts/data_inventory.py --jsons a.json b.json --image_root ...
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys

import numpy as np

TOK_PX = 1024
BOX_RE = re.compile(r"\[\s*(\d(?:\.\d+)?)\s*,\s*(\d(?:\.\d+)?)\s*,\s*(\d(?:\.\d+)?)\s*,\s*(\d(?:\.\d+)?)\s*\]")


def sample_box(s):
    """(x0, y0, x1, y1) fractions or None. Top-level bbox first; else a box in
    a human turn whose gpt turn has none (the build_tf_bbox_data convention)."""
    b = s.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        b = [float(v) for v in b]
        if b[2] > b[0] and b[3] > b[1] and max(b) <= 1.0:
            return tuple(b)
    conv = s.get("conversations") or []
    for i in range(0, len(conv) - 1, 2):
        h, g = conv[i].get("value", ""), conv[i + 1].get("value", "")
        m = BOX_RE.search(h)
        if m and not BOX_RE.search(g):
            b = [float(v) for v in m.groups()]
            if b[2] > b[0] and b[3] > b[1]:
                return tuple(b)
    return None


def image_path(s, image_root):
    im = s.get("image")
    if isinstance(im, list):
        im = im[0] if im else None
    if not im:
        return None
    return im if os.path.isabs(im) else os.path.join(image_root, im)


def image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def describe(path, image_root, n_sample, rng):
    try:
        data = json.load(open(path))
    except Exception as e:
        return f"{os.path.basename(path)}: unreadable ({e})"
    if isinstance(data, dict):
        data = data.get("data") or data.get("samples") or list(data.values())
    n = len(data)
    with_box = [(s, sample_box(s)) for s in data if sample_box(s) is not None]
    n_box = len(with_box)
    pool = with_box if with_box else [(s, None) for s in data]
    pick = rng.sample(pool, min(n_sample, len(pool)))
    long_side, box_frac, lr_px, s2_px, missing = [], [], [], [], 0
    for s, b in pick:
        p = image_path(s, image_root)
        sz = image_size(p) if p else None
        if sz is None:
            missing += 1
            continue
        W, H = sz
        long_side.append(max(W, H))
        if b is None:
            continue
        side = math.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1e-12))
        box_frac.append(side)
        # LR at 640 tokens: scale so that W*H*sc^2 = 640*1024 (never upscaled)
        sc = min(1.0, math.sqrt(640 * TOK_PX / (W * H)))
        lr_px.append(side * math.sqrt(W * H) * sc)
        # stage 2: crop of side max(4 x box side, 20 cells = 640 px) at <= 640 tokens
        crop_side = max(4 * side * math.sqrt(W * H), 640.0)
        sc2 = min(1.0, math.sqrt(640 * TOK_PX) / crop_side)
        s2_px.append(side * math.sqrt(W * H) * sc2)

    def q(a, p):
        return float(np.percentile(a, p)) if a else float("nan")
    hd_share = 100 * np.mean([v >= 1500 for v in long_side]) if long_side else float("nan")
    vstar_like = 100 * np.mean([v < 16 for v in lr_px]) if lr_px else float("nan")
    name = os.path.relpath(path, os.environ.get("OSS_DATA", "/")) if path.startswith(os.environ.get("OSS_DATA", "//")) else path
    return (f"{name}\n"
            f"    n={n:,}  with_bbox={n_box:,}  ({100 * n_box / max(n, 1):.0f}%)   images read {len(long_side)}/{len(pick)}"
            + (f"  MISSING {missing}" if missing else "") + "\n"
            f"    img long side: median {q(long_side, 50):.0f} px, >=1500 px: {hd_share:.0f}%\n"
            + (f"    box side: median {100 * q(box_frac, 50):.1f}% of image (p10 {100 * q(box_frac, 10):.1f}%)\n"
               f"    box @LR640: median {q(lr_px, 50):.0f} px (p10 {q(lr_px, 10):.0f});  @stage-2 crop: median {q(s2_px, 50):.0f} px;  "
               f"V*-like (<16 px at LR): {vstar_like:.0f}%" if box_frac else "    (no boxes)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("OSS_DATA"))
    ap.add_argument("--jsons", nargs="*", default=None)
    ap.add_argument("--image_root", default=os.path.expanduser("~/sft_data/train_split"))
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    paths = args.jsons
    if not paths:
        if not args.root:
            sys.exit("--root or --jsons")
        paths = sorted(set(glob.glob(os.path.join(args.root, "*.json"))
                           + glob.glob(os.path.join(args.root, "extra_*", "*.json"))))
    print(f"[inventory] image_root={args.image_root}  {len(paths)} json files\n")
    for p in paths:
        if os.path.getsize(p) > 6e9:
            print(f"{p}: skipped (> 6 GB)")
            continue
        print(describe(p, args.image_root, args.sample, rng))
        print()
    # image folders: what raw sources exist at all
    print("[inventory] image folders under image_root (name: #files in the top level, sample resolution):")
    if os.path.isdir(args.image_root):
        for d in sorted(os.listdir(args.image_root)):
            full = os.path.join(args.image_root, d)
            if not os.path.isdir(full):
                continue
            try:
                entries = os.listdir(full)
            except Exception:
                continue
            files = [e for e in entries if e.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            sizes = [image_size(os.path.join(full, f)) for f in rng.sample(files, min(20, len(files)))]
            sizes = [s for s in sizes if s]
            med = int(np.median([max(s) for s in sizes])) if sizes else "-"
            print(f"    {d}: {len(files)} images / {len(entries) - len(files)} subdirs  long side ~{med}")


if __name__ == "__main__":
    main()
