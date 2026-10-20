#!/usr/bin/env python3
"""Zero-GPU: how does the where-head miss on V* at the 20-cell window?

Splits the questions into "window holds the GT centre" vs "misses" and
compares target size (in LR pixels: what the head's features can see), number
of targets, question category, head confidence, and how far the head's cell is
from the target. Far misses = wrong object (data), near misses on tiny targets
= resolution (LR size / features).

  python scripts/where_miss_breakdown.py --box_json $E/loc/where_lm8_mix.json \
      --run "$E/loc/where_crop2_?.json" --dump
"""
import argparse
import glob
import json
import math
import os
import statistics as st

from PIL import Image


def load_rows(pattern):
    rows = {}
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        for r in json.load(open(f)):
            rows[r["i"]] = r
    return rows


def vstar(root):
    root = os.path.expanduser(root)
    out = {}
    for i, line in enumerate(open(os.path.join(root, "test_questions.jsonl"))):
        doc = json.loads(line)
        path = os.path.join(root, doc["image"])
        side = os.path.splitext(path)[0] + ".json"
        sd = json.load(open(side)) if os.path.isfile(side) else {}
        boxes = sd.get("bbox") or []
        if not boxes:
            continue
        W, H = Image.open(path).size
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
        out[i] = {"W": W, "H": H, "cat": str(doc.get("category", "n/a")),
                  "targets": [str(t) for t in (sd.get("target_object") or [])],
                  "boxes": boxes, "gt": (x0 / W, y0 / H, x1 / W, y1 / H),
                  "q": doc["text"].split("\n")[0]}
    return out


def inside(win, x, y):
    return win[0] <= x <= win[2] and win[1] <= y <= win[3]


def med(xs):
    return st.median(xs) if xs else float("nan")


def bucket(x, edges):
    for e in edges:
        if x < e:
            return f"<{e}"
    return f">={edges[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box_json", required=True)
    ap.add_argument("--run", required=True, help="where_crop_eval jsons of the pad-2 run")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--lr_tok", type=int, default=640)
    ap.add_argument("--dump", action="store_true", help="list every missed question")
    args = ap.parse_args()

    boxes = {r["i"]: r for r in json.load(open(os.path.expanduser(args.box_json)))}
    run = load_rows(args.run)
    gts = vstar(args.vstar_root)
    idx = sorted(set(boxes) & set(run) & set(gts))
    print(f"aligned {len(idx)}")

    items = []
    for i in idx:
        g, b, r = gts[i], boxes[i], run[i]
        W, H = g["W"], g["H"]
        lr_scale = math.sqrt(min(1.0, args.lr_tok * 1024 / float(W * H)))   # original px -> LR px
        x0, y0, x1, y1 = g["gt"]
        gcx, gcy = (x0 + x1) / 2, (y0 + y1) / 2
        # smallest single target box, in LR pixels (what the head's features see)
        min_side_lr = min(min(bb[2], bb[3]) for bb in g["boxes"]) * lr_scale
        gh, gw = b["grid"]
        hb = b["boxes"][0]
        hcx, hcy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
        dist_cells = math.hypot((hcx - gcx) * gw, (hcy - gcy) * gh)
        # distance from the head's cell to the nearest GT box edge, in cells (0 = on target)
        dx = max(x0 - hcx, 0, hcx - x1) * gw
        dy = max(y0 - hcy, 0, hcy - y1) * gh
        edge_cells = math.hypot(dx, dy)
        win = r["window"]
        items.append({
            "i": i, "cat": g["cat"], "targets": g["targets"], "q": g["q"],
            "n_t": len(g["boxes"]), "area": (x1 - x0) * (y1 - y0),
            "min_side_lr": min_side_lr, "img": f"{W}x{H}", "grid": f"{gh}x{gw}",
            "conf": b["scores"][0], "dist": dist_cells, "edge": edge_cells,
            "cov": inside(win, gcx, gcy),
            "cov_box": win[0] <= x0 and win[1] <= y0 and win[2] >= x1 and win[3] >= y1,
            "ok": bool(r["ok"]),
            # any of the top-3 cells within 3 cells of the GT box
            "top3_near": any(
                math.hypot(max(x0 - (bb[0] + bb[2]) / 2, 0, (bb[0] + bb[2]) / 2 - x1) * gw,
                           max(y0 - (bb[1] + bb[3]) / 2, 0, (bb[1] + bb[3]) / 2 - y1) * gh) <= 3
                for bb in b["boxes"]),
        })

    cov = [p for p in items if p["cov"]]
    miss = [p for p in items if not p["cov"]]
    n = len(items)
    print(f"\nwindow holds GT centre {len(cov)}/{n}; holds whole GT box {sum(p['cov_box'] for p in items)}/{n}")
    for name, part in (("covered", cov), ("missed", miss)):
        if not part:
            continue
        print(f"\n{name} ({len(part)}): acc {sum(p['ok'] for p in part)}/{len(part)}")
        print(f"  median GT area {100 * med([p['area'] for p in part]):.2f}% of image, "
              f"smallest target side {med([p['min_side_lr'] for p in part]):.0f} LR px, "
              f"targets/question {med([p['n_t'] for p in part]):.0f}, conf {med([p['conf'] for p in part]):.3f}")
        print(f"  head cell -> GT centre {med([p['dist'] for p in part]):.1f} cells, "
              f"-> GT box edge {med([p['edge'] for p in part]):.1f} cells; "
              f"top-3 within 3 cells of the box: {sum(p['top3_near'] for p in part)}/{len(part)}")
        cats = {}
        for p in part:
            cats[p["cat"]] = cats.get(p["cat"], 0) + 1
        print("  by category: " + ", ".join(f"{c} {k}" for c, k in sorted(cats.items())))

    print("\ncoverage by smallest-target side in LR px (what the head's features see)")
    for key in ("<8", "<16", "<32", "<64", ">=64"):
        part = [p for p in items if bucket(p["min_side_lr"], (8, 16, 32, 64)) == key]
        if part:
            c = sum(p["cov"] for p in part)
            print(f"  {key:>5} px: n={len(part):3d} covered {c}/{len(part)} ({100 * c / len(part):.0f}%)  "
                  f"acc {sum(p['ok'] for p in part)}/{len(part)}")

    print("\nmissed: how far the head's cell is from the GT box edge")
    for key in ("<3", "<6", "<12", ">=12"):
        part = [p for p in miss if bucket(p["edge"], (3, 6, 12)) == key]
        if part:
            print(f"  {key:>4} cells: {len(part):3d}  (acc {sum(p['ok'] for p in part)}, "
                  f"top-3 near box {sum(p['top3_near'] for p in part)}, "
                  f"median target side {med([p['min_side_lr'] for p in part]):.0f} LR px)")

    print("\nmissed by category / number of targets")
    for cat in sorted(set(p["cat"] for p in items)):
        part = [p for p in items if p["cat"] == cat]
        m = [p for p in part if not p["cov"]]
        print(f"  {cat}: {len(m)}/{len(part)} missed, acc covered "
              f"{sum(p['ok'] for p in part if p['cov'])}/{len(part) - len(m)} "
              f"missed {sum(p['ok'] for p in m)}/{len(m)}")

    if args.dump:
        print("\nmissed questions (i cat ok conf edge_cells target_side_lr img targets | question)")
        for p in sorted(miss, key=lambda p: -p["edge"]):
            print(f"  {p['i']:3d} {p['cat'][:8]:8s} ok={int(p['ok'])} conf={p['conf']:.3f} "
                  f"edge={p['edge']:5.1f} side={p['min_side_lr']:4.0f} {p['img']:>9s} "
                  f"{','.join(p['targets'])[:30]:30s} | {p['q'][:70]}")


if __name__ == "__main__":
    main()
