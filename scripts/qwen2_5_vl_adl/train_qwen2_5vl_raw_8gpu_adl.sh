#!/usr/bin/env bash
set -euo pipefail

# Single-node 8-GPU recipe for vanilla Qwen2.5-VL pretraining/fine-tuning.
# No DAT, no dual-vision inputs (single-process image path only).

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

# Use fixed large-disk root on ADL.
ADL_TMP="/root/autodl-tmp"

# Avoid numexpr thread-limit errors on high-core machines.
export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config (override by exporting env vars before running) --------
DATA_ROOT="${DATA_ROOT:-$ADL_TMP/models_data/sft_data}"
MODEL_PATH="${MODEL_PATH:-$ADL_TMP/models_data/Qwen2.5-VL-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$ADL_TMP/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$ADL_TMP/cache/vl_base}"
EXP_NAME="${EXP_NAME:-qwen2_5_vl_3b_baseline_adl}"

# Basic sanity checks to fail fast when paths are wrong.
if [[ ! -f "$DATA_ROOT/llava_hd251k.json" ]]; then
    echo "[ERROR] Missing data file: $DATA_ROOT/llava_hd251k.json" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then
    echo "[ERROR] Missing image folder: $DATA_ROOT/train_split" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing model path: $MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$CKPT_ROOT/$EXP_NAME"

# Put all compile/autotune caches on large local disk.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

# -------- Single-node 8 GPU launch config --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Single-node only: IB is not needed.
export NCCL_IB_DISABLE=1
# Keep P2P enabled by default; if your machine hangs, set NCCL_P2P_DISABLE=1.
export NCCL_P2P_DISABLE=0

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40320}" llava/train/train_qwen_dat.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen2_5_vl \
    --data_path "$DATA_ROOT/llava_hd251k.json" \
    --image_folder "$DATA_ROOT/train_split" \
    --use_dat False \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-500}" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS:-100}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --group_by_modality_length True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --seed 42 \
    --report_to "wandb" \
    --run_name "$EXP_NAME"
