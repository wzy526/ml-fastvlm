#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vldat

# 0606 Stage-2 SFT: Direction A, NO hd_gate, FULL-PARAMETER LLM.
# LAYER-COUNT ABLATION: 2 DAT layers per 6-layer module (DDLLLL x6 = 12 DAT).
# ============================================================================
#
# Design
# ------
# - Starts from the 0606 Stage-1 pretrain ckpt (Direction A, no gate, 12 DAT).
# - Direction A ON (--dat_image_hd_for_question True) — must match pretrain.
# - NO hd_gate (omit --dat_hd_gate_init). HD level governed by v_proj_hd /
#   k_proj_hd + LSE attention competition; instruction data rewards HD.
# - FULL-PARAMETER LLM (lora_enable=False, tune_mm_llm=True). The whole LLM is
#   trainable so it has maximal freedom to reorganize how it consumes the
#   (now question-visible) HD tokens — no low-rank bottleneck.
#       LLM        learning_rate = 1e-5  (conservative: full-param can wash out
#                  base general ability, which is what gives single-region tasks
#                  their headroom; bump to 2e-5 only if underfitting)
#       projector  tune_mm_mlp=True, mm_projector_lr = 1e-5
#       DAT        dat_lr = 1e-4 (smaller/newer module, needs higher lr)
#       LR ViT     frozen
#
# Output is a SELF-CONTAINED HF ckpt (full-param → no LoRA, no merge step).
#
# Memory: full-param 3B is heavier than LoRA. Default per-device batch lowered
# to 2 (grad_accum 4 → global batch 64, same as the LoRA run). Lower further
# via PER_DEVICE_BATCH / raise GRAD_ACCUM if you OOM. NOTE: 12 DAT layers (2x
# the 0604 baseline) adds memory/compute; drop PER_DEVICE_BATCH to 1 if OOM.
#
# Architecture MUST match pretrain: LSE / STE / no D1 / intention+gate ON /
# image_hd_for_question ON / no hd_gate /
# DAT_LAYERS = DDLLLLDDLLLLDDLLLLDDLLLLDDLLLLDDLLLL (12 DAT, 2 per module).
#
# Data: llava_hr_essential_sa1b_ivcap.json (369k SFT mix).

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

ADL_TMP="/root/autodl-tmp"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config --------
DATA_ROOT="${DATA_ROOT:-$ADL_TMP/models_data/sft_data}"
MODEL_PATH="${MODEL_PATH:-$ADL_TMP/vldat_experiments/0606_pretrain_sa1b_caption_dirA_nogate_12dat}"
CKPT_ROOT="${CKPT_ROOT:-$ADL_TMP/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$ADL_TMP/cache/vldat}"
EXP_NAME="${EXP_NAME:-0606_sft_dirA_nogate_full_12dat}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_hr_essential_sa1b_ivcap.json}"

if [[ ! -f "$DATA_JSON" ]]; then echo "[ERROR] Missing $DATA_JSON" >&2; exit 1; fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then echo "[ERROR] Missing $DATA_ROOT/train_split" >&2; exit 1; fi
if [[ ! -e "$DATA_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink" >&2; exit 1; fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing pretrain ckpt: $MODEL_PATH" >&2
    echo "        Run exp_pretrain_dirA_nogate_12dat.sh first." >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[ERROR] $MODEL_PATH lacks config.json (not a HF ckpt)" >&2; exit 1
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

# ABLATION: 2 DAT layers per 6-layer module (12 DAT total). Baseline = DLLLLL x6.
DAT_LAYERS="DDLLLLDDLLLLDDLLLLDDLLLLDDLLLLDDLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40951}" llava/train/train_qwen_dat.py \
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
    --dat_warmup_steps 0 \
    --dat_inject_lr_image False \
    --dat_image_hd_for_question True \
    --dat_lr 1e-4 \
    --lora_enable False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --mm_projector_lr 1e-5 \
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
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --learning_rate 1e-5 \
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

# Full-parameter run → output_dir is already a self-contained HF ckpt.
# No LoRA merge needed. Eval directly via:
#   pretrained=$CKPT_ROOT/$EXP_NAME
