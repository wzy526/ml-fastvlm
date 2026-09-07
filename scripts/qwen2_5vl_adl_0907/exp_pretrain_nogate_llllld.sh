#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"

# 0907 Stage-1 pretrain: DAT-layer PLACEMENT ablation — LLLLLD (block-END).
# ============================================================================
#
# Single-variable A/B against the proven 0701/0731 line:
#
#   history : DLLLLL x6  → D at layers  0, 6,12,18,24,30  (block start)
#   THIS run: LLLLLD x6  → D at layers  5,11,17,23,29,35  (block end)
#
# Same COUNT (6 DAT layers), same everything else (nogate, intention branch +
# as_gate ON, spatial guide OFF, grid 20, off_grps 8, inter 128, hr_scale 3,
# sa1b caption data, DAT+merger trainable @ 1e-4, LLM/ViT frozen). The ONLY
# variable is WHERE the D layers sit.
#
# Interpretation notes (decide before looking at results):
# - Block-end shifts mean D depth 15→20 AND includes the extremes: history has
#   D at layer 0 (near-embedding queries), this has D at layer 35 (injection
#   feeds only final-norm + lm_head). If the gap is large, a mid-anchored
#   control (LLDLLL) isolates the extreme-layer effect.
# - Repo prior: k/v_proj_hd consume pooler-space features ≈ shallow hidden
#   space; deeper D layers ⇒ larger domain mismatch (zero-init mitigates).
#   Prior favors block-start; if block-end wins, deep-layer retrieval beats
#   the mismatch — that's a finding, not noise.
# - Compare vs 0731 pretrain → 0817 genvs SFT → sweep, at equal LLM tokens.
#
# Consumed by: scripts/qwen2_5vl_adl_0907/exp_sft_llllld_genvs.sh

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config (OSS cluster) --------
OSS_DATA="${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/pingping.wzy}"

DATA_ROOT="${DATA_ROOT:-$OSS_DATA/sft_data}"
MODEL_PATH="${MODEL_PATH:-$OSS_DATA/Qwen2.5-VL-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$LOCAL_ROOT/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$LOCAL_ROOT/cache/vldat}"
EXP_NAME="${EXP_NAME:-0907_pretrain_nogate_llllld}"

# train_split is a SYMLINK FARM on a LOCAL fs (OSS FUSE can't create symlinks).
IMAGE_ROOT="${IMAGE_ROOT:-$LOCAL_ROOT/sft_data}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_sa1b_caption_pretrain.json}"

if [[ ! -f "$DATA_JSON" ]]; then echo "[ERROR] Missing data file: $DATA_JSON" >&2; exit 1; fi
if [[ ! -d "$IMAGE_ROOT/train_split" ]]; then echo "[ERROR] Missing $IMAGE_ROOT/train_split (create it on LOCAL disk; OSS can't hold symlinks)" >&2; exit 1; fi
if [[ ! -e "$IMAGE_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink: ln -sfn $OSS_DATA/sa1b_images $IMAGE_ROOT/train_split/sa1b" >&2; exit 1; fi
if [[ ! -d "$MODEL_PATH" ]]; then echo "[ERROR] Missing $MODEL_PATH" >&2; exit 1; fi

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

# THE experiment variable: D at block END (cf. DLLLLL x6 in 0701/0731).
DAT_LAYERS="LLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLLD"

echo "[0907-pretrain] placement ablation  dat_layers=$DAT_LAYERS  (D @ 5,11,17,23,29,35)"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40853}" llava/train/train_qwen_dat.py \
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
    --lora_enable False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --mm_projector_lr 1e-4 \
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
    --save_steps "${SAVE_STEPS:-1000}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS:-100}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --group_by_modality_length False \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --seed 42 \
    --report_to "wandb" \
    --run_name "$EXP_NAME"

# No LoRA merge needed (LLM frozen, no adapters).
# Output dir is a self-contained HF ckpt consumed by the SFT stage:
#   scripts/qwen2_5vl_adl_0907/exp_sft_llllld_genvs.sh
