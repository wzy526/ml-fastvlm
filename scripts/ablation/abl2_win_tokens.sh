#!/usr/bin/env bash
# Ablation 2, window tokens: one window of 100 / 196 / 900 tokens at LR 640.
# 400 is comp_qlas of ablation 1. The density head was trained at 400. Eval only.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for c in 100 196 900; do
    ev "tok_win$c" CROP_TOK=$c
done
say "done"
