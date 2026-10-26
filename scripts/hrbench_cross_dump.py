"""What do HR-Bench cross (FCP) questions ask, and which ones does the crop
path get wrong? Decides whether "two windows on two objects" can help at all:
if most failed cross questions are counting / whole-scene layout, no number of
peaks fixes them.

    python scripts/hrbench_cross_dump.py --psp "$E/loc/psp_mb/hr4k_*.json" --config crop --n 60
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="hrbench_4k")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--psp", default=None, help="patch_sample_probe result jsons (glob)")
    ap.add_argument("--config", default="crop")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    from probe_whd_qwen35 import _local_hrbench_parquet
    pq = args.parquet or _local_hrbench_parquet(args.split)
    cols = ["question", "answer", "category"] + list("ABCD")
    d = pd.read_parquet(pq, columns=[c for c in cols if c != "image"])
    ok = {}
    if args.psp:
        for f in glob.glob(os.path.expanduser(args.psp)):
            res = json.load(open(f))
            for r in (res.get(args.config, []) if isinstance(res, dict) else []):
                ok[r["i"]] = r["ok"]
    cross = [i for i in range(len(d)) if "cross" in str(d.iloc[i]["category"]).lower()]
    wrong = [i for i in cross if ok.get(i) is False]
    print(f"{len(cross)} cross questions; {args.config} wrong on {len(wrong)} (of {sum(i in ok for i in cross)} scored)")
    random.seed(args.seed)
    pick = random.sample(wrong or cross, min(args.n, len(wrong or cross)))
    for i in sorted(pick):
        r = d.iloc[i]
        opts = " | ".join(f"{c}. {r[c]}" for c in "ABCD" if c in r and isinstance(r[c], str))
        print(f"[{i}] {str(r['question']).strip()}  ||  {opts}  (gt {r['answer']})")


if __name__ == "__main__":
    main()
