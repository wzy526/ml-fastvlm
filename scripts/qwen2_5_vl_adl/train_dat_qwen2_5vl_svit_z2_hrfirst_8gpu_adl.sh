#!/usr/bin/env bash
set -euo pipefail

# Single-node 8-GPU recipe for Qwen2.5-VL DAT (shared-ViT, no-KD) with
# HR-first resize, hr_scale=2 variant of train_dat_qwen2_5vl_svit_z3_hrfirst_8gpu_adl.sh.
#
# Configuration:
#   - shared_vit:           True   (one HD ViT call, LR = adaptive_pool(HD))
#   - KD:                   False
#   - hr_scale:             2
#   - use_hr_first_resize:  True
#   - hr_min_pixels:        28224       (lmms-eval default)
#   - hr_max_pixels:        9031680     (lmms-eval default)
#       At hr_scale=2: HR <= 11520 tokens, derived LR <= 2880 tokens.
#   - per_device_train_batch_size 1 / gradient_accumulation_steps 8
#       Effective batch 64; LR seq ~2880 (≈ 5x of z3 LR=1280) so attention
#       compute roughly doubles vs z3 hrfirst; halving batch keeps memory
#       headroom. If OOM, lower HR_MAX_PIXELS to 6021120 (LR<=1920 tok).
#
# Step 1 of the alignment plan: same plan as z3 hrfirst, but with denser LR
# tokens. Compare against z3 hrfirst (under the same data mix) to see how the
# pool ratio (4x in z2 vs 9x in z3) trades off against absolute LR resolution.

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
EXP_NAME="${EXP_NAME:-dat_qwen2_5vl_z2_1d5l_s20_g8_i128_hd251k_lora_dat_svit_hrfirst_no_kd}"

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

# 36-layer Qwen2.5-VL-3B: 1D5L pattern, DAT on layers 0, 6, 12, 18, 24, 30.
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

# -------- HR-first resize parameters (lmms-eval aligned) ---------------------
USE_HR_FIRST="${USE_HR_FIRST:-True}"
HR_MIN_PIXELS="${HR_MIN_PIXELS:-28224}"
HR_MAX_PIXELS="${HR_MAX_PIXELS:-9031680}"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40320}" llava/train/train_qwen_dat.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen2_5_vl \
    --data_path "$DATA_ROOT/llava_hd251k.json" \
    --image_folder "$DATA_ROOT/train_split" \
    --use_hr_first_resize "$USE_HR_FIRST" \
    --hr_min_pixels "$HR_MIN_PIXELS" \
    --hr_max_pixels "$HR_MAX_PIXELS" \
    --use_dat True \
    --dat_layers "$DAT_LAYERS" \
    --dat_grid_size 20 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 2 \
    --dat_hd_proj True \
    --dat_use_intention_branch True \
    --dat_intention_as_gate True \
    --dat_shared_vit True \
    --dat_freeze_base False \
    --dat_lr 1e-4 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers "dat" \
    --lora_lr 2e-5 \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --kd_on False \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH:-1}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
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
