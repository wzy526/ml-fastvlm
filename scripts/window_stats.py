"""Native-resolution size of the sampled windows in offset_feat_train accuracy jsons.

    python scripts/window_stats.py --glob '/data/oss_bucket_0/wangziyi/vldat_experiments/ldat_eval_fix/def_*_[0-7].json'

Per dataset: the window's native HR tokens (w * h / 32^2, the tokens the ViT would make from it
unresized) against the tokens actually sampled from it, mean / median / p10 / p90.
"""
import argparse
import glob
import json
import re
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--cfg", default="dat")
    a = ap.parse_args()
    by = {}
    for f in sorted(glob.glob(a.glob)):
        m = re.search(r"(vstar|hrbench4k|hrbench8k)", f)
        rows = json.load(open(f)).get(a.cfg, [])
        for r in rows:
            for x0, y0, x1, y1 in r["window"]:
                by.setdefault(m.group(1) if m else "?", []).append(((x1 - x0) * (y1 - y0) / 1024.0,
                                                                    (x1 - x0), (y1 - y0)))
    for d, v in by.items():
        hr = sorted(t[0] for t in v)
        side = sorted((t[1] * t[2]) ** 0.5 for t in v)
        q = lambda s, p: s[min(len(s) - 1, int(p * len(s)))]
        print(f"{d:10s} n={len(v):4d}  window HR tokens: mean {st.mean(hr):7.0f}  median {q(hr, .5):7.0f}  "
              f"p10 {q(hr, .1):6.0f}  p90 {q(hr, .9):7.0f}  |  side px median {q(side, .5):5.0f}  "
              f"|  HR tokens / 400 sampled: median {q(hr, .5) / 400:.1f}x")


if __name__ == "__main__":
    main()
