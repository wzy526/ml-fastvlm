#!/usr/bin/env python3
"""Zero-GPU check: does the where-head's confidence tell a 10-cell crop that
covers the target from one that misses it?

Inputs are files that already exist:
  --box_json   where_head.py --out_boxes json (top-3 softmax probs per sample)
  --wide       where_crop_eval jsons for the pad-2 run   (77.5)
  --tight      where_crop_eval jsons for the pad-1 run   (74.3)
All three index V* the same way (enumerate over test_questions.jsonl).

Prints, per confidence quartile, how often the tight window holds the GT
centre and the accuracy of both runs, then simulates "tight if conf >= t
else wide" over every threshold. The threshold is picked on these 191
questions, so the best number is optimistic; the shape of the curve matters.

  python scripts/where_conf_check.py --box_json $E/loc/where_lm8_mix.json \
      --wide "$E/loc/where_crop2_?.json" --tight "$E/loc/where_crop_p1_?.json"
"""
import argparse
import glob
import json
import math
import os

from PIL import Image


def load_rows(pattern):
    rows = {}
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        for r in json.load(open(f)):
            rows[r["i"]] = r
    return rows


def gt_centres(root):
    root = os.path.expanduser(root)
    out = {}
    for i, line in enumerate(open(os.path.join(root, "test_questions.jsonl"))):
        doc = json.loads(line)
        path = os.path.join(root, doc["image"])
        side = os.path.splitext(path)[0] + ".json"
        boxes = (json.load(open(side)).get("bbox") or []) if os.path.isfile(side) else []
        if not boxes:
            continue
        W, H = Image.open(path).size
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
        out[i] = ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, doc.get("category", "n/a"))
    return out


def inside(win, cx, cy):
    return win[0] <= cx <= win[2] and win[1] <= cy <= win[3]


def centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def quartiles(items, key, label):
    """items sorted ascending by key, split into 4 bins; print coverage and acc."""
    items = sorted(items, key=key)
    n = len(items)
    print(f"\n{label}: low -> high")
    for q in range(4):
        part = items[q * n // 4:(q + 1) * n // 4]
        if not part:
            continue
        cov = sum(p["cov_tight"] for p in part)
        print(f"  Q{q + 1} {key(part[0]):.3f}..{key(part[-1]):.3f} n={len(part)}: "
              f"tight covers GT {cov}/{len(part)} ({100 * cov / len(part):.0f}%)  "
              f"acc tight {sum(p['ok_tight'] for p in part)}  wide {sum(p['ok_wide'] for p in part)}")


def sweep(items, key, label):
    """tight if key >= t else wide. Report the best t and the curve ends."""
    vals = sorted(set(key(p) for p in items))
    best = None
    curve = []
    for t in vals + [math.inf]:
        acc = sum(p["ok_tight"] if key(p) >= t else p["ok_wide"] for p in items)
        n_tight = sum(1 for p in items if key(p) >= t)
        curve.append((t, acc, n_tight))
        if best is None or acc > best[1]:
            best = (t, acc, n_tight)
    n = len(items)
    all_tight, all_wide = curve[0][1], curve[-1][1]
    print(f"\n{label}: all tight {all_tight}/{n}, all wide {all_wide}/{n}; "
          f"best t={best[0]:.3f} -> {best[1]}/{n} ({100 * best[1] / n:.1f}%) with {best[2]} tight")
    # a few fixed points so the shape is visible
    for frac in (0.25, 0.5, 0.75):
        t = vals[min(len(vals) - 1, int(frac * len(vals)))]
        acc = sum(p["ok_tight"] if key(p) >= t else p["ok_wide"] for p in items)
        print(f"  t={t:.3f} (top {100 * (1 - frac):.0f}% go tight): {acc}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box_json", required=True)
    ap.add_argument("--wide", required=True)
    ap.add_argument("--tight", required=True)
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    args = ap.parse_args()

    boxes = {r["i"]: r for r in json.load(open(os.path.expanduser(args.box_json)))}
    wide, tight = load_rows(args.wide), load_rows(args.tight)
    gts = gt_centres(args.vstar_root)
    idx = sorted(set(boxes) & set(wide) & set(tight) & set(gts))
    print(f"aligned {len(idx)} samples (boxes {len(boxes)}, wide {len(wide)}, tight {len(tight)})")

    items = []
    for i in idx:
        b = boxes[i]
        cx, cy, cat = gts[i]
        H, W = b["grid"]
        c0, c1 = centre(b["boxes"][0]), centre(b["boxes"][1])
        d12 = math.hypot((c0[0] - c1[0]) * W, (c0[1] - c1[1]) * H)      # cells between top-1 and top-2
        items.append({
            "i": i, "cat": cat,
            "conf": b["scores"][0],
            "gap": b["scores"][0] - b["scores"][1],
            "near": -d12,                                                # higher = peaks adjacent
            "cov_tight": inside(tight[i]["window"], cx, cy),
            "cov_wide": inside(wide[i]["window"], cx, cy),
            "ok_tight": bool(tight[i]["ok"]), "ok_wide": bool(wide[i]["ok"]),
        })

    n = len(items)
    cov_t = sum(p["cov_tight"] for p in items)
    cov_w = sum(p["cov_wide"] for p in items)
    print(f"tight window holds GT centre {cov_t}/{n} ({100 * cov_t / n:.1f}%), wide {cov_w}/{n}")
    for name, sel in (("tight covers", lambda p: p["cov_tight"]), ("tight misses", lambda p: not p["cov_tight"])):
        part = [p for p in items if sel(p)]
        if part:
            print(f"  {name} ({len(part)}): acc tight {sum(p['ok_tight'] for p in part)} "
                  f"wide {sum(p['ok_wide'] for p in part)}; mean conf "
                  f"{sum(p['conf'] for p in part) / len(part):.3f}, mean gap "
                  f"{sum(p['gap'] for p in part) / len(part):.3f}")
    oracle = sum(max(p["ok_tight"], p["ok_wide"]) for p in items)
    print(f"per-sample best of the two: {oracle}/{n} ({100 * oracle / n:.1f}%)  <- ceiling of any switch rule")

    quartiles(items, lambda p: p["conf"], "top-1 prob")
    quartiles(items, lambda p: p["gap"], "top-1 minus top-2")
    sweep(items, lambda p: p["conf"], "switch on top-1 prob")
    sweep(items, lambda p: p["gap"], "switch on top-1 minus top-2")
    sweep(items, lambda p: p["near"], "switch on top-1/top-2 adjacency")


if __name__ == "__main__":
    main()
