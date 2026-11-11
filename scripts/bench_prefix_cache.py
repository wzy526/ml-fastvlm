#!/usr/bin/env python3
"""Time one mosaic answer with and without reusing the prefix cache.

Same weights and the same questions. The cache path keeps the first forward
and only runs the mosaic plus the repeated question. Prints whether the two
answers match, and the seconds per image.
"""

import argparse
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from gdino_probe import load_vstar
from ldat_eval.infer import LDatRunner


def run(runner, samples, n_tok, flag):
    runner.prefix_cache = flag
    times, answers = [], []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for sample in samples:
        t0 = time.perf_counter()
        answers.append(runner.generate(sample["image"], sample["prompt"], n_tok))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--where_head_pt", required=True)
    ap.add_argument("--ldat_pt", required=True)
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_vstar(args.vstar_root, args.n + 1)
    if len(samples) < 2:
        raise SystemExit(f"need at least 2 V* images under {args.vstar_root}")
    runner = LDatRunner(
        args.model_path, args.where_head_pt, args.ldat_pt, device,
        crop_tok=400, lr_tok=640, n_win=1, grid=0, prefix_cache=False,
    )
    run(runner, samples[:1], args.max_new_tokens, False)
    run(runner, samples[:1], args.max_new_tokens, True)
    runner.n_cache = 0
    runner.n_cache_fallback = 0
    full_t, full_a = run(runner, samples[1:], args.max_new_tokens, False)
    cache_t, cache_a = run(runner, samples[1:], args.max_new_tokens, True)
    same = sum(a == b for a, b in zip(full_a, cache_a))
    print(f"[bench] images {len(full_a)}  answers_match {same}/{len(full_a)}")
    print(f"[bench] full  {sum(full_t) / len(full_t):.3f}s   cache {sum(cache_t) / len(cache_t):.3f}s")
    print(f"[bench] cache_used {runner.n_cache}  fallback {runner.n_cache_fallback}")
    for i, (a, b) in enumerate(zip(full_a, cache_a)):
        if a != b:
            print(f"[bench] differ {i}: {a!r} | {b!r}")


if __name__ == "__main__":
    main()
