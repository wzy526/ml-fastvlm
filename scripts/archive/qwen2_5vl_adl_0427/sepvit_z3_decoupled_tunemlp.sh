#!/usr/bin/env bash
set -euo pipefail

# Same as sepvit_z3_decoupled.sh, but with tune_mm_mlp=True so visual.merger
# is unfrozen. Ablation pair to measure whether the merger benefits from
# adapting to DAT-injected HD features under the decoupled LR/HR setup.

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
CACHE_ROOT="${CACHE_ROOT:-$ADL_TMP/cache/vldat}"
EXP_NAME="${EXP_NAME:-0427_q25vl_sepvit_z3_decoupled_tunemlp_no_kd}"

# Basic sanity checks.
if [[ ! -f "$DATA_ROOT/llava_hd251k.json" ]]; then
    echo "[ERROR] Missing data file: $DATA_ROOT/llava_hd251k.json" >&2; exit 1
fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then
    echo "[ERROR] Missing image folder: $DATA_ROOT/train_split" >&2; exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing model path: $MODEL_PATH" >&2; exit 1
fi

mkdir -p "$CKPT_ROOT/$EXP_NAME"

# Compile / autotune caches on large local disk.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

# -------- Single-node 8 GPU launch config --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# 36-layer Qwen2.5-VL-3B: 1D5L pattern, DAT on layers 0, 6, 12, 18, 24, 30.
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

# -------- Decoupled LR/HR resize parameters ----------------------------------
LR_MIN_PIXELS="${LR_MIN_PIXELS:-200704}"
LR_MAX_PIXELS="${LR_MAX_PIXELS:-501760}"
HR_MIN_PIXELS="${HR_MIN_PIXELS:-28224}"
HR_MAX_PIXELS="${HR_MAX_PIXELS:-9031680}"

# Optional dedicated LR for merger params; default reuses base learning_rate.
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40620}" llava/train/train_qwen_dat.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen2_5_vl \
    --data_path "$DATA_ROOT/llava_hd251k.json" \
    --image_folder "$DATA_ROOT/train_split" \
    --use_decoupled_hr_lr True \
    --use_hr_first_resize False \
    --lr_min_pixels "$LR_MIN_PIXELS" \
    --lr_max_pixels "$LR_MAX_PIXELS" \
    --hr_min_pixels "$HR_MIN_PIXELS" \
    --hr_max_pixels "$HR_MAX_PIXELS" \
    --use_dat True \
    --dat_layers "$DAT_LAYERS" \
    --dat_grid_size 20 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 3 \
    --dat_hd_proj True \
    --dat_use_intention_branch True \
    --dat_intention_as_gate True \
    --dat_shared_vit False \
    --dat_freeze_base False \
    --dat_lr 1e-4 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers "dat" \
    --lora_lr 2e-5 \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --mm_projector_lr "$MM_PROJECTOR_LR" \
    --kd_on False \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH:-4}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-2}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-500}" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS:-50}" \
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

# ----- Auto-merge LoRA + DAT weights into a standalone full-weight ckpt -----
source "$(dirname "${BASH_SOURCE[0]}")/_merge_after_train.sh"
