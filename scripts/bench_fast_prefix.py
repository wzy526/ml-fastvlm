#!/usr/bin/env python3
"""Check that fast_prefix gives the two-pass answer bit for bit, and time both.

Same runner and the same questions. The first loop compares the answer-turn
prefill logits and the decoded strings per image. The second loop times each
path after warmup.
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


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def answer(runner, sample, n_tok, fast):
    runner.fast_prefix = fast
    sync()
    t0 = time.perf_counter()
    text = runner.generate(sample["image"], sample["prompt"], n_tok)
    sync()
    return text, runner.last_prefill_logits, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--where_head_pt", required=True)
    ap.add_argument("--ldat_pt", required=True)
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_vstar(args.vstar_root, args.n + 1)
    if len(samples) < 2:
        raise SystemExit(f"need at least 2 V* images under {args.vstar_root}")
    runner = LDatRunner(
        args.model_path, args.where_head_pt, args.ldat_pt, device,
        crop_tok=400, lr_tok=640, n_win=1, grid=0, prefix_cache=False, fast_prefix=False,
    )
    answer(runner, samples[0], args.max_new_tokens, False)
    answer(runner, samples[0], args.max_new_tokens, True)

    same_logits = same_text = 0
    for i, sample in enumerate(samples[1:]):
        a, la, _ = answer(runner, sample, args.max_new_tokens, False)
        b, lb, _ = answer(runner, sample, args.max_new_tokens, True)
        if la is None or lb is None:
            print(f"[fast] {i} no prefill logits: {a!r} | {b!r}")
            continue
        eq = la.shape == lb.shape and torch.equal(la, lb)
        diff = (la.float() - lb.float()).abs().max().item() if la.shape == lb.shape else float("nan")
        same_logits += int(eq)
        same_text += int(a == b)
        print(f"[fast] {i} logits_equal {eq}  max_abs {diff:.4g}  {a!r} | {b!r}")
    n = len(samples) - 1
    print(f"[fast] logits_equal {same_logits}/{n}  answers_match {same_text}/{n}  "
          f"lr_vit_reused {runner.n_lr_reuse}  lr_vit_reuse_on {runner.lr_vit_reuse}")

    runner._lr_vit_checks = 0
    t_two, t_fast = [], []
    for sample in samples[1:]:
        t_two.append(answer(runner, sample, args.max_new_tokens, False)[2])
        t_fast.append(answer(runner, sample, args.max_new_tokens, True)[2])
    two, fast = sum(t_two) / n, sum(t_fast) / n
    print(f"[fast] two-pass {two:.3f}s   fast_prefix {fast:.3f}s   saved {(1 - fast / two) * 100:.1f}%")


if __name__ == "__main__":
    main()
