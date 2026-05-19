#!/usr/bin/env bash
set -euo pipefail

# Activate vldat conda env
eval "$(conda shell.bash hook)"
conda activate vldat

# Exp 11 (0519): Run D = Run C (no-warmup variant) + grid 50×50.
#
# Iteration history of this script
# --------------------------------
# v1: Run D = Run C v1 (F1 + F3 + F4 + bf16 fix) + dat_grid_size=50.
#     Run D trained ~650 steps and showed hd_gate decreasing FASTER than
#     Run C v1 (~1.7× slope), suggesting that with HD still uncalibrated,
#     a finer grid (2500 anchors vs 400) just inflates noise per HD token.
#     Stopped early.
#
# v2 (CURRENT): align with Run C v2 by dropping F3 (warmup callback);
#     keep grid=50, F1, F4, and bf16 fix.
#
# Rationale
# ---------
# The hd_gate monotonic-decline diagnosis points at warmup callback as the
# primary suspect (LoRA frozen during Phase 1 -> no positive feedback to
# integrate HD_out -> hd_gate gradient stays negative throughout). With
# warmup off, LoRA learns to consume HD_out as v_proj_hd ramps from 0.
# Once HD path can become useful end-to-end, a finer reference grid has
# a chance to actually help (more HD tokens per query = more evidence).
#
# Cost (unchanged from v1)
# ------------------------
# HD KV memory per DAT layer per sample:
#   400 -> 2500 tokens × kv_dim (256 for Qwen-3B) × 2 bytes (bf16)
#   ≈ 200 KB -> 1.3 MB per DAT layer
#   × 6 DAT layers × bs 4 × 8 GPUs ≈ 256 MB extra activation/KV memory.
# Step time grows ~15-25% over Run C (~14h -> ~16-18h).
# If OOM, set PER_DEVICE_BATCH=2 and GRAD_ACCUM=4 to preserve eff. batch.
#
# What this run keeps the same as Run C v2:
#   • Data mix
#   • F1 (--dat_use_spatial_attn_guide True)
#   • F4 (hd_input_layernorm RMSNorm fp32)
#   • bf16 round-off fix (modeling code)
#   • F3 OFF (--dat_warmup_steps 0)
#
# What this run changes vs. Run C v2:
#   • --dat_grid_size 50 (was 20)
#
# Validation question
# -------------------
# Conditional on Run C v2 recovering the V-shape hd_gate (i.e. warmup was
# the cause), does grid=50 push HR-Bench further than grid=20?
#   • Yes -> reference-density and warmup are both meaningful axes.
#   • No  -> density isn't the lever; next round must address prompt-
#            conditioning (text-token cross-attn into off_guide).

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

ADL_TMP="/root/autodl-tmp"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config --------
DATA_ROOT="${DATA_ROOT:-$ADL_TMP/models_data/sft_data}"
MODEL_PATH="${MODEL_PATH:-$ADL_TMP/models_data/Qwen2.5-VL-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$ADL_TMP/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$ADL_TMP/cache/vldat}"
EXP_NAME="${EXP_NAME:-0519_full_runD_grid50_no_warmup_F1F4_bf16fix}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_hr_essential_sa1b_ivcap.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing data file: $DATA_JSON" >&2; exit 1
fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then
    echo "[ERROR] Missing image folder: $DATA_ROOT/train_split" >&2; exit 1
fi
if [[ ! -e "$DATA_ROOT/train_split/sa1b" ]]; then
    echo "[ERROR] Missing sa1b symlink: $DATA_ROOT/train_split/sa1b" >&2; exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing model path: $MODEL_PATH" >&2; exit 1
fi

mkdir -p "$CKPT_ROOT/$EXP_NAME"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

# -------- Single-node 8 GPU --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# Full 1D5L pattern (same as Run B/C)
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40625}" llava/train/train_qwen_dat.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen2_5_vl \
    --data_path "$DATA_JSON" \
    --image_folder "$DATA_ROOT/train_split" \
    --use_hr_first_resize False \
    --hd_max_pixels 5017600 \
    --use_dat True \
    --dat_layers "$DAT_LAYERS" \
    --dat_grid_size 50 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 3 \
    --dat_hd_proj True \
    --dat_use_intention_branch True \
    --dat_intention_as_gate True \
    --dat_use_spatial_attn_guide True \
    --dat_shared_vit False \
    --dat_freeze_base False \
    --dat_hd_gate_init -1.0 \
    --dat_warmup_steps 0 \
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

# Auto-merge LoRA + DAT weights
source "$(dirname "${BASH_SOURCE[0]}")/_merge_after_train.sh"
