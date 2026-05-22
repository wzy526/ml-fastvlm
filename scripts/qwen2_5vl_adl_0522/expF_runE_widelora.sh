#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vldat

# Exp F (0521): Run E config + WIDE LoRA on ALL 36 layers' QKVO.
# ============================================================================
#
# Single-variable change vs. exp12_full_runE.sh:
#   --lora_target_layers "dat"  -->  --lora_target_layers "all"
#
# get_lora_target_modules() in train_qwen_dat.py:
#   "dat" + dat_layers="DLLLLLDLLLLL..." (6 D)  ->   6 layers ×  4 proj =  24 LoRA adapters (Run E)
#   "all"                                          -> 36 layers ×  4 proj = 144 LoRA adapters (this Run)
#
# Hypothesis being tested
# -----------------------
# All 0514+ DAT variants stuck at hrbench4k ≈ 0.58 share one invariant: the
# only LR-path-affecting trainable params are the 24-adapter LoRA on 6 DAT-layer
# QKVO. A' (LoRA all linear, no DAT) reached 0.6538 with the SAME data,
# meaning a wider LR-side LoRA scope alone buys +6 pt.
#
# Run F isolates "wider LoRA scope" without touching anything else: DAT
# structure, DAT params (D1 + D3 + intention), HD pipeline, data, all
# learning rates, all warmup are identical to Run E. Only the LoRA pattern
# string changes.
#
# Decision table for hrbench4k avg
# --------------------------------
# F ≈ A' (~0.65)     -> LR scope was the entire story; DAT structure is
#                       neutral (no help, no hurt). Recovery path: pair
#                       wider LoRA with structural fixes that connect HD
#                       to LM head (next: Run G or dense-DAT redesign).
# F ≈ base (~0.62)   -> Wider LoRA recovered most, but DAT forward graph
#                       still imposes a ~3 pt structural tax. Recovery
#                       path: identify the tax source (hd_input_layernorm
#                       numeric drift, intention_branch gate at init, ...).
# F ≈ Run E (~0.58)  -> Wider LoRA didn't help; DAT actively eats LR
#                       gains regardless of LoRA scope. Recovery path:
#                       full structural rework (HD as prefix tokens,
#                       dense injection, decoupled HD-LM-head bridge).

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

ADL_TMP="/root/autodl-tmp"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

DATA_ROOT="${DATA_ROOT:-$ADL_TMP/models_data/sft_data}"
MODEL_PATH="${MODEL_PATH:-$ADL_TMP/models_data/Qwen2.5-VL-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$ADL_TMP/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$ADL_TMP/cache/vldat}"
EXP_NAME="${EXP_NAME:-0521_expF_runE_widelora_all36qkvo}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_hr_essential_sa1b_ivcap.json}"

if [[ ! -f "$DATA_JSON" ]]; then echo "[ERROR] Missing data file: $DATA_JSON" >&2; exit 1; fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then echo "[ERROR] Missing image folder" >&2; exit 1; fi
if [[ ! -e "$DATA_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink" >&2; exit 1; fi
if [[ ! -d "$MODEL_PATH" ]]; then echo "[ERROR] Missing model path: $MODEL_PATH" >&2; exit 1; fi

mkdir -p "$CKPT_ROOT/$EXP_NAME"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# Same DAT layer pattern as Run E: 1D every 6 layers (sparse, 6 D layers).
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40741}" llava/train/train_qwen_dat.py \
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
    --dat_hd_gate_init -1.0 \
    --dat_warmup_steps 0 \
    --dat_use_residual_merge True \
    --dat_inject_lr_image True \
    --dat_lr 1e-4 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers "all" \
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

source "$(dirname "${BASH_SOURCE[0]}")/_merge_after_train.sh"
