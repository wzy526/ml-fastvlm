#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"

# 0701 Stage-2 SFT: on top of the 0701 nogate pretrain ckpt.
# ============================================================================
#
# What this is
# ------------
# Exactly the 0528 expJ SFT recipe (LoRA r=8 on LLM + unfreeze projector at
# 5e-6 + DAT trainable), but on the nogate backbone. The ONLY architectural
# change vs 0528 expJ is the absence of hd_gate:
#
#   0528 expJ : --dat_hd_gate_init -4.0   (gate inherited from pretrain)
#   0701      : (flag omitted)            → self.hd_gate = None  (NO gate)
#
# MUST match the pretrain ckpt's dat_extra_args (no gate, no dirA). The HD
# level is governed purely by the LSE attention competition; cold-start
# safety came from v_proj_hd zero-init at pretrain step 0 and is already
# baked into the source ckpt.
#
# The mm_projector fix (tune_mm_mlp=True, mm_projector_lr=5e-6) from 0528 is
# preserved: the pretrain projector ends in SA-1B "caption" mode and needs to
# drift back toward the instruction distribution during SFT.
#
# Source ckpt:   $CKPT_ROOT/0701_pretrain_sa1b_caption_fixinit_nogate/
# Data:          llava_hr_essential_sa1b_ivcap.json (369k SFT mix)
#                  — same as 0514–0528; apples-to-apples vs history.
#
# Trainable set
# -------------
# - LLM         LoRA r=8 / alpha=16 / target_layers=all / lr=2e-5
# - projector   tune_mm_mlp=True, mm_projector_lr=5e-6
# - DAT         all DAT params trainable (NO hd_gate now), lr=1e-4
# - LR ViT      frozen
#
# Architecture (MUST match the pretrain ckpt):
# - LSE merge / STE / no D1 / no D3 / no F1 / no dirA
# - intention_branch + intention_as_gate ON
# - NO hd_gate
# - DAT_LAYERS = DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config (new cluster) --------
# Data on OSS; checkpoints + caches on LOCAL fast disk. MODEL_PATH (the pretrain
# source ckpt) is derived from CKPT_ROOT so it always matches where the pretrain
# stage wrote its output.
OSS_DATA="${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/pingping.wzy}"

DATA_ROOT="${DATA_ROOT:-$OSS_DATA/sft_data}"
CKPT_ROOT="${CKPT_ROOT:-$LOCAL_ROOT/vldat_experiments}"
# Source = the 0701 nogate pretrain ckpt (written by the pretrain script).
MODEL_PATH="${MODEL_PATH:-$CKPT_ROOT/0701_pretrain_sa1b_caption_fixinit_nogate}"
CACHE_ROOT="${CACHE_ROOT:-$LOCAL_ROOT/cache/vldat}"
EXP_NAME="${EXP_NAME:-0701_expL_sft_from_fixinit_nogate_unfreeze_mlp}"

# train_split is a SYMLINK FARM on a LOCAL fs (OSS FUSE can't create symlinks,
# errno 38). JPEGs stay on OSS, reached through the symlinks. JSON stays on OSS.
IMAGE_ROOT="${IMAGE_ROOT:-$LOCAL_ROOT/sft_data}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_hr_essential_sa1b_ivcap.json}"

if [[ ! -f "$DATA_JSON" ]]; then echo "[ERROR] Missing $DATA_JSON" >&2; exit 1; fi
if [[ ! -d "$IMAGE_ROOT/train_split" ]]; then echo "[ERROR] Missing $IMAGE_ROOT/train_split (create it on LOCAL disk; OSS can't hold symlinks)" >&2; exit 1; fi
if [[ ! -e "$IMAGE_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink: ln -sfn $OSS_DATA/sa1b_images $IMAGE_ROOT/train_split/sa1b" >&2; exit 1; fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing pretrain ckpt: $MODEL_PATH" >&2
    echo "        Run exp_pretrain_sa1b_caption_fixinit_nogate.sh first." >&2
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

# HD gate: MUST match the pretrain ckpt's architecture. If the pretrain ran
# with DAT_HD_GATE_INIT=-4.0, pass the same value here, otherwise the gate
# param in the ckpt is silently dropped and the merge behaves differently.
HD_GATE_ARG=()
if [[ -n "${DAT_HD_GATE_INIT:-}" ]]; then
    HD_GATE_ARG=(--dat_hd_gate_init "${DAT_HD_GATE_INIT}")
    echo "[gate] enabling hd_gate_init=${DAT_HD_GATE_INIT}"
fi

# HD ViT early exit: MUST match the pretrain ckpt's setting (the hd adapters
# k_proj_hd/v_proj_hd learn the block-k feature distribution). Set
# DAT_HD_EARLY_EXIT_K=8/16/24 to enable; 0/unset = full depth.
EARLY_EXIT_ARG=()
if [[ -n "${DAT_HD_EARLY_EXIT_K:-}" ]]; then
    EARLY_EXIT_ARG=(--dat_hd_early_exit_k "${DAT_HD_EARLY_EXIT_K}")
    echo "[early-exit] hd_early_exit_k=${DAT_HD_EARLY_EXIT_K}"
fi

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40951}" llava/train/train_qwen_dat.py \
    "${HD_GATE_ARG[@]}" \
    "${EARLY_EXIT_ARG[@]}" \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen2_5_vl \
    --data_path "$DATA_JSON" \
    --image_folder "$IMAGE_ROOT/train_split" \
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
