#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"

# 0907 Stage-2 SFT: LLLLLD placement-ablation pretrain + the 0817 genvs mix.
# ============================================================================
#
# Consumes the stage-1 ckpt from exp_pretrain_nogate_llllld.sh and applies
# EXACTLY the 0817 genvs SFT recipe, so the only difference vs
# 0817_sft_from_nogate_retest_unfreeze_mlp_genvs is DAT layer PLACEMENT:
#
#   0817 line: DLLLLL x6 (D @ 0,6,...,30, block start)
#   THIS run : LLLLLD x6 (D @ 5,11,...,35, block end)
#
# - data: llava_hr_gen_vs_0817.json (~517k)
# - LLM LoRA r=8 a=16 lr=2e-5; projector tuned lr=5e-6; DAT lr=1e-4; ViT frozen
# - NO hd_gate / NO early-exit / NO skip-merger (matches the stage-1 ckpt)
#
# Sweep after merge and compare row-by-row against the 0817 sweep at equal
# LLM tokens (vstar/hrbench must hold; watch docvqa/chartqa/gqa direction).

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config (OSS cluster) --------
OSS_DATA="${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/pingping.wzy}"

CKPT_ROOT="${CKPT_ROOT:-$LOCAL_ROOT/vldat_experiments}"
MODEL_PATH="${MODEL_PATH:-$CKPT_ROOT/0907_pretrain_nogate_llllld}"
CACHE_ROOT="${CACHE_ROOT:-$LOCAL_ROOT/cache/vldat}"
EXP_NAME="${EXP_NAME:-0907_sft_llllld_genvs}"

# train_split is a SYMLINK FARM on a LOCAL fs (OSS FUSE can't create symlinks).
IMAGE_ROOT="${IMAGE_ROOT:-$LOCAL_ROOT/sft_data}"

DATA_JSON="${DATA_JSON:-$OSS_DATA/llava_hr_gen_vs_0817.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing $DATA_JSON (build via construct_sft_0817.py)" >&2; exit 1
fi
if [[ ! -d "$IMAGE_ROOT/train_split" ]]; then echo "[ERROR] Missing $IMAGE_ROOT/train_split (create it on LOCAL disk; OSS can't hold symlinks)" >&2; exit 1; fi
if [[ ! -e "$IMAGE_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink: ln -sfn $OSS_DATA/sa1b_images $IMAGE_ROOT/train_split/sa1b" >&2; exit 1; fi
# All prefixes referenced by the 0817 mix must be symlinked into the local farm.
for prefix in stvqa deepeyes visualprobe densefusion allava aokvqa scienceqa; do
    if [[ ! -e "$IMAGE_ROOT/train_split/$prefix" ]]; then
        echo "[ERROR] Missing $prefix symlink: ln -sfn $OSS_DATA/train_split/$prefix $IMAGE_ROOT/train_split/$prefix" >&2
        exit 1
    fi
done
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing stage-1 pretrain ckpt: $MODEL_PATH" >&2
    echo "        Run exp_pretrain_nogate_llllld.sh first (or point MODEL_PATH at the OSS copy)." >&2
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

# MUST match stage 1 (dat_extra_args is rebuilt from CLI, not read from ckpt).
DAT_LAYERS="LLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLLD"

echo "[0907-sft] placement ablation  dat_layers=$DAT_LAYERS  data=$(basename "$DATA_JSON")"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40977}" llava/train/train_qwen_dat.py \
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
source "$(dirname "${BASH_SOURCE[0]}")/../qwen2_5vl_adl_0701/_merge_after_train.sh"
