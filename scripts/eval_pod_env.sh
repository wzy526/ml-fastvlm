#!/usr/bin/env bash
# Pod-side env for lmms-eval. Source this before eval_pixel_sweep.sh:
#   source /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/eval_pod_env.sh
#   bash scripts/eval_pixel_sweep.sh dat35 "$CKPT" chartqa tag
#
# Data is on the offline-2 volume; code is on x2v-2. Pods typically mount both.

export EVAL_OFFLINE=1
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MMBENCH_SKIP_GPT_EVAL=1

# Platform injects NCCL_DEBUG=INFO. Use NCCL_DEBUG_LEVEL=INFO to get it back.
export NCCL_DEBUG="${NCCL_DEBUG_LEVEL:-WARN}"

export HF_HOME="${HF_HOME:-/home/ea-cv-nlp-train-offline-2/xzf/hf_eval}"
export HF_EVAL_ROOT="${HF_EVAL_ROOT:-$HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

export LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-/home/ea-cvfa-aigc-x2v-2/xzf/lmms-eval}"
export VLDAT_VENV="${VLDAT_VENV:-/home/ea-cvfa-aigc-x2v-2/xzf/venvs/vldat}"
# `llava` (DAT modeling) is only pip-installed in some envs; put the repo root on
# PYTHONPATH so the qwen*_dat wrappers import regardless.
export FASTVLM_DIR="${FASTVLM_DIR:-/home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm}"
export LMMS_FASTVLM_PATH="${LMMS_FASTVLM_PATH:-$FASTVLM_DIR}"
export PYTHONPATH="${LMMS_EVAL_DIR}:${FASTVLM_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# InfoSeek images (optional; jsonl is already under extra/infoseek/)
export INFOSEEK_IMAGE_ROOT="${INFOSEEK_IMAGE_ROOT:-$HF_HOME/extra/infoseek/images}"

# Do not let HF libraries try the public hub / wandb.
unset HF_ENDPOINT HF_HUB_ENABLE_HF_TRANSFER 2>/dev/null || true

echo "[eval_pod_env] HF_HOME=$HF_HOME"
echo "[eval_pod_env] LMMS_EVAL_DIR=$LMMS_EVAL_DIR"
echo "[eval_pod_env] FASTVLM_DIR=$FASTVLM_DIR"
echo "[eval_pod_env] VLDAT_VENV=$VLDAT_VENV"
