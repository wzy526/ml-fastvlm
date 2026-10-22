#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vldat

# Exp I (0527): SFT on top of the BAD-init 0526 pretrain ckpt.
# ============================================================================
#
# What this run is for
# --------------------
# 0526 pretrain (`0526_pretrain_sa1b_caption_lse_ste`) was trained with the
# conv_off_proj Kaiming-init bug -- offsets started ≈ 1.4 std → saturated
# at the image border within the first ~50 steps, never recovered.
# A clean fix-init pretrain is queued separately under
# `qwen2_5vl_adl_0526/exp13_pretrain_sa1b_caption.sh` with EXP_NAME
# `0526_pretrain_sa1b_caption_lse_ste_fixinit`.
#
# Meanwhile, this run answers a different question:
#   "If we run our previous SFT recipe on the BAD pretrain ckpt,
#    does the projector that was trained on 500k SA-1B captions
#    survive into hrbench numbers, or do we converge back to the
#    0514–0520 'bug ckpt cluster' of hb4k≈0.58?"
#
# The diagnostic value is:
#   - Result ≈ base (0.62) ............ SA-1B caption pretrain helped the
#                                       projector enough to undo the bug
#                                       hit; DAT module is being ignored
#                                       (frozen LLM perspective).
#   - Result ≈ 0.58 (bug cluster) ..... Bug is dominant: SFT cannot
#                                       recover offsets that have already
#                                       collapsed during pretrain → must
#                                       wait for fix-init pretrain.
#   - Result > 0.62 .................. SA-1B caption pretrain + this SFT
#                                       beats base even with bug, which
#                                       would be a strong signal that
#                                       projector quality matters more
#                                       than DAT offset quality. Unlikely
#                                       but worth checking.
#
# Architecture decisions (locked, MUST match the pretrain ckpt)
# -------------------------------------------------------------
# - LSE merge .................... yes (modeling default)
# - STE on sample_locs ........... yes (modeling default; clamp fwd, identity bwd)
# - D1 dat_inject_lr_image ....... OFF (matches pretrain)
# - D3 use_residual_merge ........ REMOVED from code path
# - F1 spatial_attn_guide ........ OFF (matches pretrain; do NOT enable
#                                   here -- pretrain ckpt has no F1 weights
#                                   to inherit, would init-from-scratch
#                                   and corrupt continuity)
# - hd_gate ...................... NONE (matches pretrain; do NOT set
#                                   --dat_hd_gate_init -- pretrain ckpt
#                                   has no hd_gate Parameter to load)
# - DAT_LAYERS ................... DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL
#                                   (sparse 6D, same as pretrain & 0522/0523)
# - dat_grid_size / off_grps /
#   inter_size / hr_scale / ...... all unchanged from pretrain
#
# Trainable parts (this is the SFT delta)
# ---------------------------------------
# - LLM .......... LoRA r=8 / alpha=16 / all layers / lr=2e-5
# - DAT .......... trainable, lr=1e-4 (continues from pretrain init)
# - Projector .... FROZEN here (tune_mm_mlp False), since pretrain already
#                  trained it on SA-1B captions. Matches 0523 expH pattern.
# - LR ViT ....... frozen (tune_mm_vision False)
#
# Data
# ----
# Same SFT mix as 0514–0523: llava_hr_essential_sa1b_ivcap.json
# (~369k samples). Mixed instruction-following + sa1b vqa.

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

ADL_TMP="/root/autodl-tmp"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config --------
DATA_ROOT="${DATA_ROOT:-$ADL_TMP/models_data/sft_data}"
# Source: the 0526 BAD-init pretrain ckpt.
MODEL_PATH="${MODEL_PATH:-$ADL_TMP/vldat_experiments/0526_pretrain_sa1b_caption_lse_ste}"
CKPT_ROOT="${CKPT_ROOT:-$ADL_TMP/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$ADL_TMP/cache/vldat}"
EXP_NAME="${EXP_NAME:-0527_expI_sft_from_0526pretrain_bad}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_hr_essential_sa1b_ivcap.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing data file: $DATA_JSON" >&2
    exit 1
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
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[ERROR] $MODEL_PATH does not look like a HF ckpt (no config.json)" >&2; exit 1
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

# Same DAT layer pattern as pretrain & 0522/0523: 1D every 6 layers.
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40831}" llava/train/train_qwen_dat.py \
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

# Merge LoRA + non-LoRA trainables (DAT params, etc.) into a self-contained
# HF ckpt at $CKPT_ROOT/$EXP_NAME-merged for downstream lmms-eval.
source "$(dirname "${BASH_SOURCE[0]}")/_merge_after_train.sh"
