#!/usr/bin/env python3
"""Zero-GPU: would a second (third) window from the head cover the targets the
top-1 window misses?  Uses the top-3 boxes the head wrote into the box json and
the exact top-1 window size from a where_crop_eval run (same head, same
geometry), so cover@1 here must reproduce the run's coverage.
  python scripts/where_topk_cover.py --box_json $E/loc/where_lm8_mix.json \
      --run "$E/loc/p1/zero_train_?.json"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from where_miss_breakdown import inside, load_rows, vstar  # noqa: E402


def window_like(box, ref_win):
    ww, wh = ref_win[2] - ref_win[0], ref_win[3] - ref_win[1]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    cx = min(max(cx, ww / 2), 1 - ww / 2)
    cy = min(max(cy, wh / 2), 1 - wh / 2)
    return cx - ww / 2, cy - wh / 2, cx + ww / 2, cy + wh / 2


def overlap(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box_json", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--acc_cov", type=float, default=85.0, help="run accuracy when the window covers the target")
    ap.add_argument("--acc_unc", type=float, default=58.8, help="... when it misses")
    args = ap.parse_args()
    boxes = {r["i"]: r for r in json.load(open(os.path.expanduser(args.box_json)))}
    run = load_rows(args.run)
    gt = vstar(args.vstar_root)
    keys = sorted(set(boxes) & set(run) & set(gt))
    print(f"{len(keys)} questions")
    cov = [0, 0, 0]
    ov12, dist2, cats = [], [], {}
    second_hits = []
    for i in keys:
        g = gt[i]["gt"]
        cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        ref = run[i]["window"]
        wins = [ref] + [window_like(b, ref) for b in boxes[i]["boxes"][1:3]]
        hit = [inside(w, cx, cy) for w in wins]
        for k in range(3):
            cov[k] += any(hit[:k + 1])
        ov12.append(overlap(wins[0], wins[1]))
        if not hit[0]:
            c = cats.setdefault(run[i]["cat"], [0, 0])
            c[0] += 1
            c[1] += hit[1]
            if hit[1]:
                second_hits.append(i)
            b1, b2 = boxes[i]["boxes"][0], boxes[i]["boxes"][1]
            dist2.append(((b1[0] - b2[0]) ** 2 + (b1[1] - b2[1]) ** 2) ** 0.5)
    n = len(keys)
    print(f"GT centre covered: top-1 {cov[0]}/{n} ({100 * cov[0] / n:.1f})  "
          f"top-1|2 {cov[1]}/{n} ({100 * cov[1] / n:.1f})  top-1|2|3 {cov[2]}/{n} ({100 * cov[2] / n:.1f})")
    print(f"window-2 overlaps window-1 by (fraction of area): mean {sum(ov12) / n:.2f}, "
          f">0.5 in {sum(o > 0.5 for o in ov12)}/{n}  (large = second crop is nearly the same view)")
    print(f"among the {n - cov[0]} top-1 misses: window-2 covers {cov[1] - cov[0]}; "
          f"median |box1-box2| {sorted(dist2)[len(dist2) // 2] if dist2 else float('nan'):.2f} of the image side")
    for c, (m, h) in sorted(cats.items()):
        print(f"   {c}: misses {m}, rescued by window-2 {h}")
    gain = (cov[1] - cov[0]) * (args.acc_cov - args.acc_unc) / 100
    print(f"expected gain if a second crop behaved like a covered one: ~{gain:.1f} questions "
          f"({100 * gain / n:+.1f} pt), ids {second_hits}")


if __name__ == "__main__":
    main()
