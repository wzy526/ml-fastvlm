#!/usr/bin/env python3
"""Zero-GPU: two where_crop_eval runs side by side, split by whether the
window held the GT centre (coverage). Tells "LM got better at reading the
crop" (covered acc up) apart from "LM started trusting the crop" (uncovered
acc down) -- the 0919 over-trust pattern.
  python scripts/where_run_diff.py --a "$E/loc/p1/zero_train_?.json" \
      --b "$M/eval/head_?.json" --label_a before --label_b after
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from where_miss_breakdown import inside, load_rows, vstar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    args = ap.parse_args()
    A, B = load_rows(args.a), load_rows(args.b)
    gt = vstar(args.vstar_root)
    keys = sorted(set(A) & set(B))
    print(f"{len(keys)} common questions ({len(A)} / {len(B)} in the runs)")

    def cov(r, i):
        g = gt.get(i)
        if g is None:
            return None
        cx, cy = (g["gt"][0] + g["gt"][2]) / 2, (g["gt"][1] + g["gt"][3]) / 2
        return inside(r["window"], cx, cy)

    groups = {"covered(A)": [], "uncovered(A)": [], "covered(B)": [], "uncovered(B)": []}
    same_win = 0
    for i in keys:
        ca, cb = cov(A[i], i), cov(B[i], i)
        if ca is None:
            continue
        groups["covered(A)" if ca else "uncovered(A)"].append(i)
        groups["covered(B)" if cb else "uncovered(B)"].append(i)
        same_win += ca == cb
    print(f"coverage: {args.label_a} {len(groups['covered(A)'])}, {args.label_b} {len(groups['covered(B)'])}; "
          f"same coverage status {same_win}/{len(keys)}")
    for name in ("covered(B)", "uncovered(B)"):
        ids = groups[name]
        oa = sum(A[i]["ok"] for i in ids)
        ob = sum(B[i]["ok"] for i in ids)
        print(f"{name:14s} n={len(ids):3d}  {args.label_a} {oa:3d} ({100 * oa / max(len(ids), 1):.1f})  "
              f"{args.label_b} {ob:3d} ({100 * ob / max(len(ids), 1):.1f})  delta {ob - oa:+d}")
    gained = [i for i in keys if B[i]["ok"] and not A[i]["ok"]]
    lost = [i for i in keys if A[i]["ok"] and not B[i]["ok"]]
    print(f"flips: gained {len(gained)} {gained}; lost {len(lost)} {lost}")
    for tag, ids in (("gained", gained), ("lost", lost)):
        c = sum(1 for i in ids if cov(B[i], i))
        print(f"  {tag}: window covered the target in {c}/{len(ids)}")
    cats = sorted({A[i]["cat"] for i in keys})
    for c in cats:
        ids = [i for i in keys if A[i]["cat"] == c]
        print(f"  {c:20s} n={len(ids):3d} {args.label_a} {sum(A[i]['ok'] for i in ids)} "
              f"{args.label_b} {sum(B[i]['ok'] for i in ids)}")


if __name__ == "__main__":
    main()
