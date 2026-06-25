#!/usr/bin/env bash
# Unattended overnight runner: full 6-pixel DAT sweep across the benchmark suite.
# Default = ALL benchmarks EXCEPT multi-image ones (MMMU is excluded — DAT's
# multi-image HD path is untrained, so MMMU is left to a dedicated run).
# VQAv2 is also excluded (val ~214k, won't finish overnight).
#
# Multi-image feature is explicitly OFF here (DAT_MULTI_IMAGE=0): every single-
# image benchmark is bit-identical to the pre-multi-image code path.
#
# Two phases, both fault-tolerant (one bad task never aborts the night):
#   Phase 1 (warm):  single-process `--limit` run per task to download + verify
#                    its dataset cache (avoids the multi-rank download race that
#                    corrupted the textvqa cache). Retries up to 3x per task.
#   Phase 2 (sweep): per task, the full 6-pixel DAT sweep via eval_pixel_sweep.sh
#                    (which has its own per-point done-markers + failure guard).
#
# Usage:
#   nohup bash scripts/overnight_general_sweep.sh [CKPT] [TAG] > overnight.log 2>&1 &
#
# Resumable: re-running skips any task/pixel point that already has a `done` marker.

set -uo pipefail

CKPT="${1:-/root/autodl-tmp/vldat_experiments/0528_expJ_sft_from_fixinit_unfreeze_mlp-merged}"
TAG="${2:-0528_expJ}"
BASE_REF="${BASE_REF:-/root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct}"
SWEEP="/root/autodl-tmp/ml-fastvlm/scripts/eval_pixel_sweep.sh"

# Task list: override with TASKS_OVERRIDE="t1 t2 ..." (space-separated).
# Default = all benchmarks EXCEPT multi-image (MMMU). 7 high-res + 7 general.
if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
    read -r -a TASKS <<< "$TASKS_OVERRIDE"
else
    TASKS=(
        docvqa_val chartqa gqa textvqa_val vstar_bench hrbench4k hrbench8k
        scienceqa_img vizwiz_vqa_val pope mme mmbench_en_dev mmbench_cn_dev seedbench
    )
fi

eval "$(conda shell.bash hook)"
conda activate vldat
cd /root/autodl-tmp/lmms-eval

# Mirror for downloads. Do NOT set TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE: recent
# huggingface_hub treats them as a global offline switch that also blocks
# `datasets` from fetching uncached benchmarks (this is what made every warmup
# fail with ConnectionError earlier). The model loads from a local dir → no hub
# call, so offline mode is unnecessary.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT=1200
export NUMEXPR_MAX_THREADS=64
unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE HF_DATASETS_OFFLINE 2>/dev/null || true

# Multi-image DAT HD path is OFF for this run (multi-image samples — if any
# slipped in — fall back to plain causal attention; single-image is unchanged).
export DAT_MULTI_IMAGE=0

echo "############################################################"
echo "# overnight general sweep   ckpt=$CKPT  tag=$TAG"
echo "# tasks: ${TASKS[*]}"
echo "# start: $(date)"
echo "############################################################"

# ---- Phase 1: warm dataset caches via direct load_dataset (proven path) ----
# lmms-eval's own download wrapper was flaky here; a plain datasets.load_dataset
# over the mirror reliably warms the cache, which lmms-eval then reads offline.
# Loads ALL splits per (path, config) so split-name guessing isn't needed.
echo; echo "===== PHASE 1: dataset warmup (direct load_dataset) ====="
python3 - << 'PYEOF'
import sys, time
from datasets import load_dataset, get_dataset_config_names

# (path, config) — config None means default
specs = [
    ("lmms-lab/ScienceQA", "ScienceQA-IMG"),
    ("lmms-lab/VizWiz-VQA", None),
    ("lmms-lab/POPE", None),
    ("lmms-lab/MME", None),
    ("lmms-lab/MMBench", "en"),
    ("lmms-lab/MMBench", "cn"),
    ("lmms-lab/SEED-Bench", None),
]
# NOTE: MMMU is intentionally NOT warmed/evaluated here (multi-image; excluded).
# High-res tasks (docvqa/chartqa/gqa/textvqa/vstar/hrbench) are already cached
# from earlier sweeps, so Phase 2 reads them directly.

for path, cfg in specs:
    label = f"{path}" + (f":{cfg}" if cfg else "")
    for attempt in (1, 2, 3):
        try:
            t0 = time.time()
            ds = load_dataset(path, cfg) if cfg else load_dataset(path)
            n = sum(len(s) for s in ds.values())
            print(f"[warm OK] {label}  ({n} rows, {time.time()-t0:.0f}s)", flush=True)
            break
        except Exception as e:
            print(f"[warm retry {attempt}] {label}: {type(e).__name__}: {e}", flush=True)
            time.sleep(15)
    else:
        print(f"[WARN] warmup FAILED for {label} after 3 tries", flush=True)
print("PHASE1_DONE", flush=True)
PYEOF

# ---- Phase 2: full 6-pixel DAT sweep per task -----------------------------
echo; echo "===== PHASE 2: 6-pixel DAT sweep ====="
for t in "${TASKS[@]}"; do
    echo; echo "########## SWEEP $t  $(date +%H:%M:%S) ##########"
    bash "$SWEEP" dat "$CKPT" "$t" "$TAG" || echo "[FAIL] sweep $t aborted — continuing" >&2
done

echo; echo "############################################################"
echo "# ALL DONE: $(date)"
echo "# results under: /root/autodl-tmp/ml-fastvlm/_test_outputs/_sweep_<task>_${TAG}/"
echo "############################################################"
