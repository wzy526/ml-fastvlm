#!/usr/bin/env bash
set -euo pipefail

# Activate vldat conda env
eval "$(conda shell.bash hook)"
conda activate vldat

# Exp A' (0521): "LoRA-only baseline on caption mix, NO DAT at all."
# ============================================================================
#
# Diagnostic purpose
# ------------------
# Run E (0520) shows DAT ckpts ~7-13 pt below the vanilla Qwen2.5-VL-3B-Instruct
# baseline on HR-Bench, even with the HD path explicitly silenced at eval time
# (HD-OFF in _diagnose_dat_alive.py). That means the LR path of the DAT ckpt
# has been corrupted by training, not just "HD did nothing".
#
# Two confounded suspects in Run E's recipe degrade the LR path:
#   (S1) LoRA on QKVO of the 6 DAT layers (L0/L6/L12/L18/L24/L30).
#   (S2) ~369K SA-1B-style short captions that don't reward fine-grained
#        understanding, but do shift the LLM toward caption-style outputs.
#
# This Run isolates (S1 ∪ S2) WITHOUT (DAT itself):
#   • model:  base Qwen2.5-VL-3B-Instruct (no DAT class, no DAT params)
#   • data:   llava_hr_essential_sa1b_ivcap.json (identical to Run E)
#   • LoRA:   enabled, r=8, alpha=16, lr=2e-5, no QLoRA, no dropout
#   • LoRA target_modules: find_all_linear_names(model)
#       -> q/k/v/o + gate/up/down in ALL 36 LLM layers (252 adapters).
#       Note: Run E LoRA was QKVO on 6 layers only (24 adapters), so this
#       is a STRICTLY BROADER LoRA scope. If even this scope does NOT
#       degrade HR-Bench vs. base, then Run E's degradation is owned by
#       the DAT params themselves, not LoRA+caption. If this scope DOES
#       degrade roughly as much as Run E (~7-13 pt drop), then LoRA on
#       attention projections + caption data is the primary culprit, and
#       DAT is at most a secondary contributor.
#   • tune_mm_*: all False  ->  PEFT freezes everything except LoRA adapters.
#   • use_dat: False  ->  uses Qwen2VLSupervisedDataset, no HD path
#       (hd_max_pixels / dat_* / lora_target_layers are all ignored).
#
# Hparams that matter for fairness with Run E
# -------------------------------------------
#   • learning_rate 2e-5,  warmup 50,  cosine,  max_grad_norm 1.0,  bf16
#   • per_device_train_batch_size 4,  grad_accum 2,  8 GPUs
#       -> effective batch 64, same as Run E.
#   • num_train_epochs 1,  data ~369K  ->  ~5770 steps.
#   • seed 42 (matches Run E).
#
# Eval plan after this trains
# ---------------------------
# Run lmms_eval with the SAME pareto config Run E used (max_pixels=9031680,
# use_decoupled_hr_lr True, lr_min_pixels/lr_max_pixels matching Run E eval).
# Compare against:
#   • base Qwen2.5-VL-3B-Instruct (already known scores)
#   • Run E (0520) DAT ckpt
#   • This A' ckpt
# On hrbench_4k + hrbench_8k (DocVQA optional).
#
# Outputs
# -------
# Adapter dir: $CKPT_ROOT/$EXP_NAME
# Merged dir : $CKPT_ROOT/$EXP_NAME-merged   (via _merge_after_train.sh)

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
EXP_NAME="${EXP_NAME:-0521_expA_prime_lora_all_no_dat}"

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

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40731}" llava/train/train_qwen_dat.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen2_5_vl \
    --data_path "$DATA_JSON" \
    --image_folder "$DATA_ROOT/train_split" \
    --use_hr_first_resize False \
    --use_dat False \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
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

# Auto-merge LoRA weights into a flat HF ckpt for lmms-eval
source "$(dirname "${BASH_SOURCE[0]}")/_merge_after_train.sh"
