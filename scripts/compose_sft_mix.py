#!/usr/bin/env python3
"""Compose an SFT training json from a base mix plus bbox side sets.

Replaces the ad-hoc --mix flags scattered over the bbox builders. Each --add
can be repeated (oversampling), filtered by `source`, and capped; the result is
shuffled with a fixed seed and a per-source / bbox summary is printed so the
composition is on record in the launch log.

0920 combined run:
  OSS=/data/oss_bucket_0/wangziyi/models_data
  python scripts/compose_sft_mix.py \
      --base $OSS/llava_hr_gen_vs_0817.json \
      --add $OSS/extra_0916/viscot_bbox.train.json \
      --add $OSS/synth_hd_text_50k.json --only_source synth_hd --max_n 30000 \
      --add $OSS/extra_0920/viscot_nat_bbox.json \
      --out $OSS/llava_hr_gen_vs_0817_bbox0920.json
  (viscot_bbox.train.json = viscot_bbox.json minus the 500 held-out used by the
   probe / leverage tests -- keep using the .train split so those stay clean;
   synth_hd_text_50k.json also contains 50k base samples mixed in at build
   time, --only_source synth_hd keeps just the synthetic ones; the 1500x2250
   SA-1B backgrounds are the only natural-image boxes that pass the HD/LR
   guard, see build_viscot_nat_bbox_data.py.)

Every sample keeps its fields; `source` is filled from --tag when missing
(base samples usually carry none -> tagged "base").
"""
import argparse
import collections
import json
import os
import random


def load(path):
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, list):
        raise SystemExit(f"{path}: expected a list, got {type(d).__name__}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--add", action="append", default=[], help="side set json (repeatable)")
    ap.add_argument("--repeat", action="append", type=int, default=[],
                    help="per --add: repeat factor (default 1; align by position)")
    ap.add_argument("--only_source", action="append", default=[],
                    help="per --add: keep only samples whose `source` starts with this ('' = all)")
    ap.add_argument("--max_n", action="append", type=int, default=[],
                    help="per --add: random cap after filtering (0 = no cap)")
    ap.add_argument("--tag", default="base", help="`source` for samples that carry none")
    ap.add_argument("--require_bbox", type=int, default=1,
                    help="1 = a side set must contain bbox samples (guards against passing the wrong json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    def opt(lst, i, default):
        return lst[i] if i < len(lst) else default

    base = load(args.base)
    for s in base:
        s.setdefault("source", args.tag)
    print(f"base {args.base}: {len(base)}")
    mixed = list(base)

    for i, path in enumerate(args.add):
        rep, only, cap = opt(args.repeat, i, 1), opt(args.only_source, i, ""), opt(args.max_n, i, 0)
        d = load(path)
        n0 = len(d)
        if only:
            d = [s for s in d if str(s.get("source", "")).startswith(only)]
        if args.require_bbox and not any(s.get("bbox") for s in d):
            raise SystemExit(f"{path}: no bbox samples after filter '{only}' -- wrong file?")
        if cap and len(d) > cap:
            rng.shuffle(d)
            d = d[:cap]
        nb = sum(1 for s in d if s.get("bbox"))
        for s in d:
            s.setdefault("source", os.path.splitext(os.path.basename(path))[0])
        for k in range(rep):
            for s in d:
                y = dict(s)
                if k:
                    y["id"] = f"{s.get('id', '')}_r{k}"
                mixed.append(y)
        print(f"add  {path}: {n0} rows -> kept {len(d)} (filter '{only}', cap {cap or '-'}; bbox {nb}) x{rep}")

    rng.shuffle(mixed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(mixed, f, ensure_ascii=False)

    by_src = collections.Counter(s["source"] for s in mixed)
    nbox = sum(1 for s in mixed if s.get("bbox"))
    print(f"\nwrote {len(mixed)} -> {args.out}")
    print(f"bbox samples: {nbox} ({nbox / len(mixed):.1%} -> expected dat/tf_frac)")
    for src, n in by_src.most_common():
        nb = sum(1 for s in mixed if s["source"] == src and s.get("bbox"))
        print(f"  {src:32s} {n:8d} ({n / len(mixed):5.1%})" + (f"  bbox {nb}" if nb else ""))
    pref = collections.Counter(str(s.get("image", "")).split("/")[0] for s in mixed if s.get("image"))
    print("image prefixes (each must be linked into train_split): " +
          ", ".join(f"{k}:{v}" for k, v in pref.most_common()))


if __name__ == "__main__":
    main()
