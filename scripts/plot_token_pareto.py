#!/usr/bin/env python3
"""Accuracy-vs-LLM-visual-tokens Pareto plot from eval_pixel_sweep.sh outputs.

Each sweep dir (OUT_ROOT of eval_pixel_sweep.sh) contains one subdir per
point, named <model>_tok<N>, holding lmms-eval's *_results.json. This script
reads any number of such dirs, extracts one metric, and draws one curve per
dir on a log-x token axis.

Usage:
  python scripts/plot_token_pareto.py --task hrbench4k --metric average \
      --out pareto_hr4k.png \
      "Qwen3.5-2B=_test_outputs/_sweep_hrbench4k_base35_2b" \
      "Qwen3.5-2B-DAT=_test_outputs/_sweep_hrbench4k_0826_q35" \
      "DAT (HD off)=_test_outputs/_sweep_hrbench4k_0826_q35_hdoff"

  --metric  average | single | cross  (hrbench); for other tasks pass the
            metric name without the ',none' suffix.
"""

import argparse
import glob
import json
import os
import re


def load_curve(root, task, metric):
    pts = []
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        m = re.search(r"tok(\d+)", os.path.basename(d.rstrip("/")))
        if not m:
            continue
        tok = int(m.group(1))
        for f in glob.glob(os.path.join(d, "**", "*_results.json"), recursive=True):
            try:
                res = json.load(open(f))["results"][task]
            except Exception:
                continue
            v = res.get(f"{metric},none")
            if isinstance(v, (int, float)):
                pts.append((tok, v * 100 if v <= 1.0 else v))
                break
    pts.sort()
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("curves", nargs="+", help='"Label=/path/to/sweep_dir" ...')
    ap.add_argument("--task", required=True)
    ap.add_argument("--metric", default="average")
    ap.add_argument("--out", default="pareto.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    markers = ["o", "s", "^", "D", "v", "P"]
    print(f"{'label':>18} | " + " ".join(f"{'tok':>6}:{'acc':>6}" for _ in range(1)))
    for i, spec in enumerate(args.curves):
        label, root = spec.split("=", 1)
        pts = load_curve(root, args.task, args.metric)
        if not pts:
            print(f"[warn] no points for {label} in {root}")
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=markers[i % len(markers)], label=label, lw=1.8, ms=5)
        for x, y in pts:
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7)
        print(f"{label:>18} | " + " ".join(f"{x:>6}:{y:6.2f}" for x, y in pts))

    ax.set_xscale("log", base=2)
    ax.set_xlabel("LLM visual tokens per image")
    ax.set_ylabel(f"{args.task} {args.metric} (%)")
    ax.set_title(args.title or f"{args.task}: accuracy vs. LLM visual tokens")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
