"""Split the random-mode sheet of vis_hrbench_sampling.py into rows and lay each category out in columns.

    python scripts/vis_split_rows.py --dir /tmp/vis_supp --n_show 8 --cols 2

Rows are separated by runs of at least --min_gap white pixel rows. Random mode stacks the rows per
category in sorted order, correct answers first, which is how rows are matched to hrbench_sampling.json.
Writes <dir>/fig_<category>.jpg and <dir>/row_<category>_<k>_ok<0|1>.jpg.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def split_rows(img, min_gap, thresh=250):
    a = np.asarray(img.convert("L"))
    blank = (a >= thresh).all(axis=1)
    rows, start, run = [], None, 0
    for y, b in enumerate(blank):
        if b:
            run += 1
            if start is not None and run >= min_gap:
                rows.append((start, y - run + 1))
                start = None
        else:
            if start is None:
                start = y
            run = 0
    if start is not None:
        rows.append((start, len(blank)))
    out = []
    for y0, y1 in rows:
        band = a[y0:y1]
        xs = np.where(~(band >= thresh).all(axis=0))[0]
        out.append(img.crop((max(0, xs[0] - 8), max(0, y0 - 8), min(img.width, xs[-1] + 9), min(img.height, y1 + 8))))
    return out


def grid(rows, cols, gap=24):
    w = max(r.width for r in rows)
    n = (len(rows) + cols - 1) // cols
    heights = [max(r.height for r in rows[i * cols:(i + 1) * cols]) for i in range(n)]
    canvas = Image.new("RGB", (cols * w + gap * (cols - 1), sum(heights) + gap * (n - 1)), (255, 255, 255))
    y = 0
    for i in range(n):
        for j, r in enumerate(rows[i * cols:(i + 1) * cols]):
            canvas.paste(r, (j * (w + gap), y))
        y += heights[i] + gap
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/vis_supp")
    ap.add_argument("--n_show", type=int, default=8)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--min_gap", type=int, default=26)
    args = ap.parse_args()

    sheet = Image.open(os.path.join(args.dir, "hrbench_sampling.png")).convert("RGB")
    with open(os.path.join(args.dir, "hrbench_sampling.json")) as f:
        meta = json.load(f)
    order = []
    for cat in sorted({m["category"] for m in meta}):
        ms = [m for m in meta if m["category"] == cat]
        order += sorted(ms, key=lambda m: not m["correct"])[:args.n_show]
    rows = split_rows(sheet, args.min_gap)
    if len(rows) != len(order):
        raise SystemExit(f"split {len(rows)} rows but the json lists {len(order)}; try another --min_gap")
    for cat in sorted({m["category"] for m in order}):
        picked = [(r, m) for r, m in zip(rows, order) if m["category"] == cat]
        for k, (r, m) in enumerate(picked):
            r.save(os.path.join(args.dir, f"row_{cat}_{k}_ok{int(m['correct'])}.jpg"), quality=92)
        fig = grid([r for r, _m in picked], args.cols)
        fig.save(os.path.join(args.dir, f"fig_{cat}.jpg"), quality=92)
        n_ok = sum(m["correct"] for _r, m in picked)
        print(f"{cat}: {n_ok}/{len(picked)} correct, fig_{cat}.jpg {fig.width}x{fig.height}")
        for k, (_r, m) in enumerate(picked):
            print(f"  {k} ok={int(m['correct'])} pred={m['pred'][:8]!r} gt={m['answer']} | {m['question'][:80]}")


if __name__ == "__main__":
    main()
