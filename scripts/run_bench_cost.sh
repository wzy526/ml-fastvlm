#!/usr/bin/env bash
# Cost rows of the paper table for one backbone on one GPU:
# Native (LR), Native (HR), SFT control (LR), Q-LAS (pixel sampling, fast_prefix).
#
#   SIZE=4b|2b  [OUT=dir]  [GPU=0]  [MAX_SAMPLES=50]  [DATASETS="vstar hrbench4k hrbench8k"]
#   SMOKE=1     two V* rows per method, to check the counters before a full run
#   LMMS_EVAL_DIR=~/lmms-eval-oldvstar   take the task prompts from that checkout
#
# Latency is only meaningful on an otherwise idle machine.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE="${SIZE:-4b}"
X="${X:-/data/oss_bucket_0/wangziyi/vldat_experiments/sa1b_ldat}"
OFF="${OFF:-/data/oss_bucket_0/wangziyi/official_ckpt}"
if [[ "$SIZE" == "4b" ]]; then
    NATIVE="$OFF/Qwen3.5-4B"
    SFT="$HOME/vldat_experiments/1110_4b_viscot_lora-merged"
    WHERE="$SFT/where_lm8_ce20.pt"
    LDAT="$SFT/ldat_zero.pt"
elif [[ "$SIZE" == "2b" ]]; then
    NATIVE="$OFF/Qwen3.5-2B"
    SFT="$HOME/vldat_experiments/0924_miniE1_2k/checkpoint-1000"
    WHERE="$HOME/vldat_experiments/0924_miniE1_2k/loc/where_lm8_ce20.pt"
    LDAT="$X/hinge_noratio_ldat.pt"
else
    echo "SIZE must be 4b or 2b" >&2
    exit 1
fi
for p in "$NATIVE" "$SFT" "$WHERE" "$LDAT"; do
    [[ -e "$p" ]] || { echo "[ERROR] missing $p" >&2; exit 1; }
done

OUT="${OUT:-$X/cost_$SIZE}"
N="${MAX_SAMPLES:-50}"
DS="${DATASETS:-vstar hrbench4k hrbench8k}"
WARM="${WARMUP:-2}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    N=2; DS=vstar; WARM=1
fi
if [[ -n "${LMMS_EVAL_DIR:-}" ]]; then
    export PYTHONPATH="${LMMS_EVAL_DIR}${PYTHONPATH:+:$PYTHONPATH}"
fi
mkdir -p "$OUT"
echo "[cost] SIZE=$SIZE OUT=$OUT samples=$N datasets=$DS"

run() {
    CUDA_VISIBLE_DEVICES="${GPU:-0}" python scripts/bench_cost.py --name "$1" --model "$2" --model_args "$3" \
        --datasets $DS --max_samples "$N" --warmup "$WARM" --out "$OUT/cost.jsonl"
}
run "Native (LR)" qwen_native "pretrained=$NATIVE,max_tok=640"
run "Native (HR)" qwen_native "pretrained=$NATIVE,max_tok=-1"
run "SFT control (LR)" qwen_native "pretrained=$SFT,max_tok=640"
run "Q-LAS" ldat "pretrained=$SFT,where_head_pt=$WHERE,ldat_pt=$LDAT,fast_prefix=True"

python scripts/bench_cost_table.py "$OUT/cost.jsonl" --datasets $DS
