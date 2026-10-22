#!/usr/bin/env bash
set -euo pipefail

# Activate vldat conda env
eval "$(conda shell.bash hook)"
conda activate vldat

# Exp 10 (0519): Run C = bf16 round-off fix + F1 + F4, NO F3 (warmup callback).
#
# Iteration history of this script
# --------------------------------
# v1: Run C = Run B config + offset-path bf16 round-off fix (F1 + F3 + F4).
#     Run C trained ~1000 steps and showed hd_gate monotonically decreasing
#     throughout (no V-shape recovery), even past step 500 where F3's Phase
#     1 -> Phase 2 transition was expected to flip hd_gate's gradient sign.
#     Compared to the earlier zero-init-only experiment, which showed a
#     down-then-up hd_gate trajectory, the only differing variables here
#     are F1 + F3 + F4 + bf16-fix.
#
# v2 (CURRENT): drop F3 to isolate whether warmup callback is the cause of
#     the monotonic hd_gate decline. Keep F1 + F4 + bf16 fix.
#
# Hypothesis for v2
# -----------------
# During Phase 1 (LoRA frozen) of F3:
#   • v_proj_hd starts zero-init, k_proj_hd kaiming. HD_out = 0 initially.
#   • k_proj_hd still siphons attention mass via the LSE merge -> "wasted"
#     attention on V=0 -> loss wants smaller w_HD -> hd_gate gradient
#     negative -> hd_gate falls.
#   • LoRA is frozen, so the LLM cannot adapt to integrate HD_out as
#     v_proj_hd starts learning a non-zero direction. The HD signal stays
#     in "noise" from the LLM's perspective. There is no positive feedback
#     to flip hd_gate's gradient sign.
#   • By the time Phase 2 unfreezes LoRA, hd_gate has already trained
#     into a "HD is bad" prior; the trajectory continues downward.
#
# Without F3 (this run): LoRA trains alongside DAT from step 0. v_proj_hd
# learning a useful direction + LoRA learning to consume it form a positive
# feedback loop, mirroring the original zero-init experiment's down-then-
# up pattern.
#
# Background on the bf16 fix (kept from Run C v1)
# -----------------------------------------------
# In Run B, ``conv_off_proj.weight`` (≈ 0.05), ``ln_2.weight`` (= 1.0),
# ``ln_1.weight`` (= 1.0), ``proj_intention.weight``, ``conv_lr_*.weight``
# were stored in bf16. AdamW per-step update at dat_lr=1e-4 is ≈ 1e-4,
# at or below the bf16 grid spacing at these magnitudes (2⁻⁷ ≈ 7.8e-3 at
# weight ≈ 1.0; 2⁻¹¹ ≈ 5e-4 at weight ≈ 0.05). bf16 downcast silently
# rounded every update back -> offset network never learned across 1 full
# Run B epoch.
#
# Fix (in llava/model/language_model/modeling_qwen2_5vl_dat.py):
#   • New _FP32WeightLayerNorm2d / _FP32WeightConv2d / _FP32WeightLinear
#     store ``weight`` (and ``bias``) in fp32; forward downcasts to input
#     dtype.
#   • Qwen2_5_VLAttentionDAT._apply and convert_qwen2_5vl_to_dat enforce
#     fp32 on conv_lr_dw, ln_1, conv_lr_proj, proj_intention, ln_2,
#     conv_off_proj, plus the existing hd_gate / hd_input_layernorm.weight.
#
# What this run keeps the same as Run B:
#   • Data mix (llava_hr_essential_sa1b_ivcap.json)
#   • F1 (--dat_use_spatial_attn_guide True)
#   • F4 (hd_input_layernorm RMSNorm, now fp32 + actually trains)
#   • All optim / LoRA / data hyperparams
#
# What this run changes vs. Run C v1:
#   • F3 disabled (--dat_warmup_steps 0). LoRA trains from step 0
#     alongside DAT.
#
# Expected wandb signatures (vs. Run C v1)
# ----------------------------------------
# Primary: dat/hd_gate_raw should show a V-shape (decrease early, then
# recover). The recovery slope is the key signal — if it materializes,
# warmup callback was the cause of the prior monotonic decline.

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
EXP_NAME="${EXP_NAME:-0519_full_runC_no_warmup_F1F4_bf16fix}"

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

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40623}" llava/train/train_qwen_dat.py \
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
