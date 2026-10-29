"""Summarise run_lora3.sh: each LoRA version scored on its own training config, plus paired sign tests.

    python scripts/lora3_summary.py [--dir /data/oss_bucket_0/wangziyi/vldat_experiments/lora3]

Reads {version}_{dataset}_{shard}.json (offset_feat_train accuracy mode). Rows are LoRA versions,
columns every config that version was scored with; the config it was trained on is marked with *.
Head-to-head compares the three versions on their own config, paired by question index.
"""
import argparse
import glob
import json
import math
import os

VERSIONS = {"net": "net", "fzero": "fzero", "crop": "crop"}   # version -> config it was trained on
DATASETS = ["vstar", "hrbench4k", "hrbench8k"]
CFGS = ["crop", "fpool", "fzero", "net"]


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(d, version, dataset):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, f"{version}_{dataset}_*.json"))):
        for c, lst in json.load(open(f)).items():
            for r in lst:
                rows.setdefault(c, {})[r["i"]] = bool(r["ok"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/data/oss_bucket_0/wangziyi/vldat_experiments/lora3")
    args = ap.parse_args()
    res = {(v, ds): load(args.dir, v, ds) for v in VERSIONS for ds in DATASETS}

    for ds in DATASETS:
        print(f"\n=== {ds}: accuracy (correct/n), * = config the LoRA was trained on")
        print(f"{'LoRA version':<14}" + "".join(f"{c:>16}" for c in CFGS))
        for v, own in VERSIONS.items():
            r = res[(v, ds)]
            if not r:
                print(f"{v:<14}  (no results yet)")
                continue
            cells = []
            for c in CFGS:
                if c not in r:
                    cells.append(f"{'-':>16}")
                    continue
                k, n = sum(r[c].values()), len(r[c])
                cells.append(f"{('*' if c == own else '') + f'{k}/{n} {100 * k / max(n, 1):.1f}%':>16}")
            print(f"{v:<14}" + "".join(cells))

        own = {v: res[(v, ds)].get(c) for v, c in VERSIONS.items() if res[(v, ds)].get(c)}
        names = list(own)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                x, y = own[names[a]], own[names[b]]
                common = sorted(set(x) & set(y))
                w = sum(x[i] and not y[i] for i in common)
                l = sum(y[i] and not x[i] for i in common)
                print(f"  LoRA-{names[a]} vs LoRA-{names[b]} (own configs, n={len(common)}): "
                      f"{w} win / {l} lose  p={sign_p(w, l):.3f}")


if __name__ == "__main__":
    main()
