#!/usr/bin/env bash
set -euo pipefail

# Activate vldat conda env
eval "$(conda shell.bash hook)"
conda activate vldat

# All-D-layers experiment: every LLM layer gets a DAT module.
#
# Baseline uses 1D5L pattern (6 DAT layers out of 36).
# This experiment tests whether making ALL 36 layers DAT layers
# can train successfully and whether the additional capacity helps.
#
# Key differences from baseline_lrfirst.sh:
#   - DAT_LAYERS: all 36 chars are "D" (vs 1D5L pattern)
#   - Significantly more trainable parameters (6x more DAT modules)
#   - May need lower learning rate for stability
#   - Memory/throughput impact: ~6x more DAT params per forward pass
#
# Risks:
#   - OOM on 8×GPU with batch=4 (try batch=2 + grad_accum=4 if so)
#   - Training instability from too many new random modules
#   - Slower convergence if signal-to-noise ratio drops

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
EXP_NAME="${EXP_NAME:-0430_all_d_layers}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_hr_essential_350k.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing data file: $DATA_JSON" >&2
    echo "       Run: python scripts/qwen2_5vl_adl_0430/build_hr_essential_mix.py --verify_images" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then
    echo "[ERROR] Missing image folder: $DATA_ROOT/train_split" >&2; exit 1
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

# ALL 36 layers are DAT layers
DAT_LAYERS="DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40530}" llava/train/train_qwen_dat.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen2_5_vl \
    --data_path "$DATA_JSON" \
    --image_folder "$DATA_ROOT/train_split" \
    --use_hr_first_resize False \
    --hd_max_pixels 5017600 \
    --use_dat True \
    --dat_layers "$DAT_LAYERS" \
    --dat_grid_size 20 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 3 \
    --dat_hd_proj True \
    --dat_use_intention_branch True \
    --dat_intention_as_gate True \
    --dat_use_spatial_attn_guide False \
    --dat_shared_vit False \
    --dat_freeze_base False \
    --dat_lr 5e-5 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers "dat" \
    --lora_lr 1e-5 \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --kd_on False \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH:-2}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-500}" \
    --learning_rate 1e-5 \
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

# Auto-merge LoRA + DAT weights
source "$(dirname "${BASH_SOURCE[0]}")/_merge_after_train.sh"
