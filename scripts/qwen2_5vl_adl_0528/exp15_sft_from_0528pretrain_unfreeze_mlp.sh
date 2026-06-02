#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vldat

# Exp 15 (0528): SFT on top of the FIXED 0528 pretrain ckpt.
# ============================================================================
#
# Key change vs. 0527 expI (post-mortem fix)
# ------------------------------------------
# (★) tune_mm_mlp = TRUE  +  mm_projector_lr = 5e-6   ← THE FIX
#     0527 expI inherited 0523 expH's recipe with tune_mm_mlp=False,
#     which made sense when there was no pretrain step (projector starts
#     from base, already instruction-ready). But with a pretrain that
#     trained projector on 500k SA-1B captions, the projector ends in
#     "caption description" mode (long-form scene description style).
#     SFT with frozen projector → LLM LoRA cannot un-caption-mode the
#     visual features → OCRBench drops 10 pt vs from-base baseline.
#     LLaVA-standard Stage-2: pretrain freezes LLM, SFT unfreezes both.
#     5e-6 (vs pretrain's 1e-4) lets projector slowly drift back toward
#     instruction distribution without erasing the HD alignment learned
#     in pretrain.
#
# Source ckpt:   $CKPT_ROOT/0528_pretrain_sa1b_caption_fixinit_hdgate-4/
# Data:          llava_hr_essential_sa1b_ivcap.json (369k SFT mix)
#                  — exactly what 0514–0523 used; apples-to-apples vs history.
#
# Trainable set
# -------------
# - LLM         LoRA r=8 / alpha=16 / target_layers=all / lr=2e-5
# - projector   tune_mm_mlp=True, mm_projector_lr=5e-6   ← THE FIX
# - DAT         all DAT params trainable (incl. hd_gate), lr=1e-4
# - LR ViT      frozen
#
# Architecture (MUST match the pretrain ckpt's dat_extra_args):
# - LSE merge / STE / no D1 / no D3 / no F1 / intention_branch+gate ON
# - hd_gate_init -4.0  (inherited from pretrain, continues to learn)
# - DAT_LAYERS = DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

ADL_TMP="/root/autodl-tmp"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config --------
DATA_ROOT="${DATA_ROOT:-$ADL_TMP/models_data/sft_data}"
# Source = the FIXED 0528 pretrain ckpt.
MODEL_PATH="${MODEL_PATH:-$ADL_TMP/vldat_experiments/0528_pretrain_sa1b_caption_fixinit_hdgate-4}"
CKPT_ROOT="${CKPT_ROOT:-$ADL_TMP/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$ADL_TMP/cache/vldat}"
EXP_NAME="${EXP_NAME:-0528_expJ_sft_from_fixinit_unfreeze_mlp}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_hr_essential_sa1b_ivcap.json}"

if [[ ! -f "$DATA_JSON" ]]; then echo "[ERROR] Missing $DATA_JSON" >&2; exit 1; fi
if [[ ! -d "$DATA_ROOT/train_split" ]]; then echo "[ERROR] Missing $DATA_ROOT/train_split" >&2; exit 1; fi
if [[ ! -e "$DATA_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink" >&2; exit 1; fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing pretrain ckpt: $MODEL_PATH" >&2
    echo "        Run exp14_pretrain_sa1b_caption_fixinit_hdgate-4.sh first." >&2
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

DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40931}" llava/train/train_qwen_dat.py \
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
    --dat_hd_gate_init -4.0 \
    --dat_warmup_steps 0 \
    --dat_inject_lr_image False \
    --dat_lr 1e-4 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers "all" \
    --lora_lr 2e-5 \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --mm_projector_lr 5e-6 \
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
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
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

# Auto-merge LoRA + non-LoRA trainables (DAT params + projector deltas)
# into a self-contained HF ckpt at $CKPT_ROOT/$EXP_NAME-merged.
source "$(dirname "${BASH_SOURCE[0]}")/_merge_after_train.sh"
