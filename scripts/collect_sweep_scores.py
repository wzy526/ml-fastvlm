#!/usr/bin/env python3
"""Collect lmms-eval sweep scores into one table.

Walks _test_outputs/_sweep_<task>_<tag>_<arm>/<kind>_tok<N>/<ckpt>/*_results.json
and prints accuracy per arm per token budget, so arms can be compared at a
matched visual-token budget.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict


def pick_metric(res):
    """Return (name, value) for the first real accuracy-like metric."""
    out = {}
    for task, m in res.get("results", {}).items():
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if not isinstance(v, (int, float)):
                continue
            if k.endswith("_stderr") or k == "alias" or "samples" in k:
                continue
            out[f"{task}/{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser(
        "~/ml-fastvlm/_test_outputs"))
    ap.add_argument("--pattern", default="_sweep_hrbench4k_0906_res_*")
    ap.add_argument("--metric", default=None,
                    help="substring of the metric key, e.g. 'average'. "
                         "Default picks the most widely populated one.")
    args = ap.parse_args()

    # arm -> token -> {metric: value}
    table = defaultdict(dict)
    tokens = set()
    for d in sorted(glob.glob(os.path.join(args.root, args.pattern))):
        arm = os.path.basename(d).split("_res_", 1)[-1] \
            if "_res_" in os.path.basename(d) else os.path.basename(d)
        for pt in sorted(glob.glob(os.path.join(d, "*_tok*"))):
            m = re.search(r"tok(\d+)", os.path.basename(pt))
            if not m:
                continue
            tok = int(m.group(1))
            js = sorted(glob.glob(os.path.join(pt, "*", "*_results.json")))
            if not js:
                continue
            try:
                res = json.load(open(js[-1]))
            except Exception as e:
                print("  ! unreadable %s: %s" % (js[-1], e))
                continue
            mets = pick_metric(res)
            if mets:
                table[arm][tok] = mets
                tokens.add(tok)

    if not table:
        print("no results under %s/%s" % (args.root, args.pattern))
        return

    tokens = sorted(tokens)
    # the metric present in the most cells wins the main table
    counts = defaultdict(int)
    for arm in table:
        for tok in table[arm]:
            for k in table[arm][tok]:
                counts[k] += 1
    if args.metric:
        cand = [k for k in counts if args.metric in k]
        if not cand:
            print("no metric matching %r; have: %s"
                  % (args.metric, ", ".join(sorted(counts))))
            return
        main_metric = max(cand, key=counts.get)
    else:
        main_metric = max(counts, key=counts.get)

    print("metric: %s   (%d/%d cells)" %
          (main_metric, counts[main_metric], len(table) * len(tokens)))
    print()
    hdr = "%-16s" % "arm" + "".join("%10s" % ("tok%d" % t) for t in tokens)
    print(hdr)
    print("-" * len(hdr))
    for arm in sorted(table):
        row = "%-16s" % arm
        for t in tokens:
            v = table[arm].get(t, {}).get(main_metric)
            row += "%10s" % ("%.4f" % v if v is not None else "-")
        print(row)

    others = [k for k in counts if k != main_metric]
    if others:
        print("\nother metrics present: " + ", ".join(sorted(others)[:8]))


if __name__ == "__main__":
    main()
