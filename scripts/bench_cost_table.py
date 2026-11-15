#!/usr/bin/env python
"""Rows for the cost columns from scripts/bench_cost.py output.

Tokens V/G (k), FLOPs (T) and Latency (s) are averaged over the datasets
listed in --datasets (each dataset's mean counts once, like Avg.3). A later
line with the same --name replaces an earlier one.

    python scripts/bench_cost_table.py $OUT/cost.jsonl
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--datasets", nargs="+", default=["hrbench4k", "hrbench8k", "vstar"])
    ap.add_argument("--detail", action="store_true", help="also print every dataset and the breakdowns")
    args = ap.parse_args()

    runs = {}
    with open(args.jsonl) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                runs[r["name"]] = r

    print(f"{'Method':<24} {'V (k)':>7} {'G (k)':>7} {'FLOPs (T)':>10} {'Latency (s)':>12}   datasets")
    for name, r in runs.items():
        ds = [d for d in args.datasets if d in r["datasets"]]
        if not ds:
            continue

        def avg(key):
            vals = [r["datasets"][d].get(key) for d in ds]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        v, g, fl, lat = avg("visual_tokens"), avg("gen_tokens"), avg("flops_T"), avg("latency_s")
        fl_s = f"{fl:10.2f}" if fl is not None else f"{'-':>10}"
        missing = [d for d in args.datasets if d not in ds]
        note = ",".join(ds) + (f"  (missing {','.join(missing)})" if missing else "")
        print(f"{name:<24} {v / 1e3:7.2f} {g / 1e3:7.3f} {fl_s} {lat:12.2f}   {note}")
        if args.detail:
            for d in ds:
                s = r["datasets"][d]
                print(f"    {d:<10} n={s['n']:<4} lat {s['latency_s']:.2f}s (p50 {s['latency_s_p50']:.2f})  "
                      f"V {s['visual_tokens']:.0f}  G {s['gen_tokens']:.1f}  "
                      f"phases(ms) {{{', '.join(f'{k} {x:.0f}' for k, x in s['phase_ms'].items())}}}  "
                      f"gpu(ms) {{{', '.join(f'{k} {x:.0f}' for k, x in s['group_ms'].items())}}}")
                if "flops_T" in s:
                    print(f"               FLOPs {s['flops_T']:.2f}T  "
                          f"phases {{{', '.join(f'{k} {x:.2f}' for k, x in s['phase_flops_T'].items())}}}  "
                          f"groups {{{', '.join(f'{k} {x:.2f}' for k, x in s['group_flops_T'].items())}}}  "
                          f"gdn_est {s['gdn_est_T']:.3f}T  linear_check {s['linear_check_min']:.3f}")

    print()
    print("LaTeX (Tokens V/G (k) & FLOPs (T) & Latency (s)):")
    for name, r in runs.items():
        ds = [d for d in args.datasets if d in r["datasets"]]
        if not ds:
            continue
        vals = {k: [r["datasets"][d].get(k) for d in ds] for k in ("visual_tokens", "gen_tokens", "flops_T", "latency_s")}
        m = {k: (sum(x for x in v if x is not None) / len([x for x in v if x is not None])
                 if any(x is not None for x in v) else None) for k, v in vals.items()}
        fl = f"{m['flops_T']:.2f}" if m["flops_T"] is not None else "--"
        print(f"{name} & {m['visual_tokens'] / 1e3:.2f}/{m['gen_tokens'] / 1e3:.3f} & {fl} & {m['latency_s']:.2f} \\\\")


if __name__ == "__main__":
    main()
