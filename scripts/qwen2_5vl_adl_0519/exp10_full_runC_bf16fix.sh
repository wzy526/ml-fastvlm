#!/usr/bin/env bash
set -euo pipefail

# Activate vldat conda env
eval "$(conda shell.bash hook)"
conda activate vldat

# Exp 10 (0519): Run C = Run B config + offset-path bf16 round-off fix.
#
# Background
# ----------
# Run B (0515) added F1 (spatial_attn_guide) + F3 (DAT-only warmup 500 steps)
# + F4 (hd_input_layernorm RMSNorm before k/v_proj_hd) on top of the
# zero-init baseline. Eval after Run B showed essentially no improvement
# over the prior zero-init run, even though structural changes were sound.
#
# Diagnosis (scripts/qwen2_5vl_adl_0515/_diagnose_offsets*.py):
#   • Offset behavior in Run B was virtually identical to a kaiming-random
#     init DAT (built via _build_random_dat_ckpt.py).
#   • Cause: ``conv_off_proj.weight`` (≈ 0.05), ``ln_2.weight`` (= 1.0),
#     ``ln_1.weight`` (= 1.0), ``proj_intention.weight``, ``conv_lr_*.weight``
#     are all stored in bf16. AdamW per-step update at dat_lr=1e-4 is ≈ 1e-4,
#     which is at or below the bf16 grid spacing at these weight magnitudes
#     (2⁻⁷ ≈ 7.8e-3 at weight ≈ 1.0; 2⁻¹¹ ≈ 5e-4 at weight ≈ 0.05). The bf16
#     downcast silently rounded every update back to the same value -> offset
#     network never actually learned across 1 full epoch of Run B.
#
# Fix (in llava/model/language_model/modeling_qwen2_5vl_dat.py):
#   • New _FP32WeightLayerNorm2d / _FP32WeightConv2d / _FP32WeightLinear
#     subclasses store ``weight`` (and ``bias``) in fp32. Forward downcasts
#     to input dtype, so compute remains bf16-cheap; storage retains
#     full-precision update accumulation.
#   • Qwen2_5_VLAttentionDAT._apply and convert_qwen2_5vl_to_dat now
#     explicitly enforce fp32 storage on every conv_lr_dw, ln_1,
#     conv_lr_proj, proj_intention, ln_2, conv_off_proj parameter (plus
#     the existing hd_gate and hd_input_layernorm.weight protections).
#
# Validation (200-step smoke run, 0519_smoke_offset_fp32_runB_cfg_200steps):
#   • _diagnose_offsets shows L6 magn_mean_avg goes from 0.192 (Run B,
#     1 epoch) -> 0.476 (smoke, 200 steps) — i.e. 200 steps of properly-
#     stored training did MORE to offset weights than 1 full Run B epoch.
#   • Cross-sample std L6/L18/L30 jumped by +60% / +225% / +150% vs
#     random init — offset network is genuinely learning, no longer just
#     drifting around init.
#   • _verify_offset_fp32.py confirms AdamW |Δw| ≈ 2e-4 per step on the
#     target params (was ≈ 0 under bf16 storage).
#
# Caveat (still TODO):
#   • Prompt-conditional diagnostic shows offsets still don't follow prompt
#     direction (agree = 1/4 across all layers). F1's spatial_attn_guide is
#     image-saliency-only; question/prompt info is not reaching the offset
#     predictor in a directional way. If Run C's HR-Bench eval is still
#     flat vs. Run B, the next iteration should replace F1 with explicit
#     text-token cross-attention into the offset predictor.
#
# What this run keeps the same as Run B:
#   • Data mix (llava_hr_essential_sa1b_ivcap.json)
#   • F1 (--dat_use_spatial_attn_guide True)
#   • F3 (--dat_warmup_steps 500, two-phase DAT-then-LoRA schedule)
#   • F4 (hd_input_layernorm RMSNorm, now fp32 + actually trains)
#   • All optim / LoRA / data hyperparams
#
# What this run changes:
#   • Picks up the offset-path fp32 protection in modeling code
#   • New EXP_NAME / output_dir / master_port
#
# Expected wandb signatures (vs. Run B 0515):
#   • dat/grad_norm comparable to Run B (the gradients were always there;
#     it was the storage that was discarding them).
#   • dat/kvhd_weight_norm should keep climbing — same as Run B.
#   • offset stats logged inside _generate_offsets_and_sample (offset.mean
#     /.std) should show meaningfully larger drift over training than Run B.
#   • train/loss curve should be roughly comparable to Run B early on, then
#     diverge as the HD path actually contributes.

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
EXP_NAME="${EXP_NAME:-0519_full_runC_bf16fix_F1F3F4}"

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

# Full 1D5L pattern (same as Run B)
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40619}" llava/train/train_qwen_dat.py \
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
    --dat_use_spatial_attn_guide True \
    --dat_shared_vit False \
    --dat_freeze_base False \
    --dat_hd_gate_init -1.0 \
    --dat_warmup_steps 500 \
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
