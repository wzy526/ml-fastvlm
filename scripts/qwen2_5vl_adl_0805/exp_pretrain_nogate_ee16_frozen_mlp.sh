#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"

# 0805 Stage-1 pretrain: SA-1B caption, nogate backbone + HD ViT EARLY EXIT k=16,
# projector (mm MLP) FROZEN.
# ============================================================================
#
# What this is
# ------------
# Same recipe as the proven 0701/0731 nogate pretrain with exactly TWO changes:
#
#   1. HD early exit k=16: the HD branch's ViT runs only the first 16 of 32
#      blocks (~half the HD ViT FLOPs). Includes full-attention blocks 7 and 15,
#      so the truncated stack still has global mixing. The from-scratch hd
#      adapters (hd_input_layernorm + k/v_proj_hd) learn the block-16 feature
#      distribution natively — no train/test mismatch at eval.
#   2. Projector FROZEN (tune_mm_mlp=False): visual.merger stays at base
#      Qwen2.5-VL weights for the whole run. Trainable set = DAT module ONLY.
#      (0701 trained the merger at lr 1e-4 during pretrain.)
#
#   NOTE: freezing the merger does NOT hurt the early-exit adaptation — the HD
#   path runs under torch.no_grad(), so the merger never receives HD-branch
#   gradients anyway; block-16 adaptation lives entirely in the DAT hd adapters.
#
# Everything else identical to 0701 (= 0528 exp14 backbone, no hd_gate, no
# dirA): LSE merge / STE / intention_branch + intention_as_gate / sparse
# 1D-per-6 DAT layers / grid 20 / off_grps 8 / inter 128 / hr_scale 3 / hd 5M.
#
# Trainable set (LLM frozen for Stage-1 alignment)
# ------------------------------------------------
# - DAT module (conv_off_proj, k/v_proj_hd, intention branch)   lr 1e-4
# - visual.merger (projector)                                    FROZEN  <- diff vs 0701
# - LLM / LR ViT                                                 FROZEN
#
# Data: llava_sa1b_caption_pretrain.json (503k SA-1B captions from InternVL).

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config (new cluster) --------
OSS_DATA="${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/pingping.wzy}"

DATA_ROOT="${DATA_ROOT:-$OSS_DATA/sft_data}"
MODEL_PATH="${MODEL_PATH:-$OSS_DATA/Qwen2.5-VL-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$LOCAL_ROOT/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$LOCAL_ROOT/cache/vldat}"
EXP_NAME="${EXP_NAME:-0805_pretrain_nogate_ee16_frozen_mlp}"

# train_split is a SYMLINK FARM on a LOCAL fs (OSS FUSE can't create symlinks).
IMAGE_ROOT="${IMAGE_ROOT:-$LOCAL_ROOT/sft_data}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_sa1b_caption_pretrain.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing data file: $DATA_JSON" >&2
    echo "        Build it first via:" >&2
    echo "          python scripts/qwen2_5vl_adl_0430/build_sa1b_caption_pretrain.py" >&2
    exit 1
fi
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

# Sparse 1D-per-6 layer pattern (same as 0528/0701).
DAT_LAYERS="DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"

# HD ViT early exit — THE POINT of this experiment. Default 16; override with
# DAT_HD_EARLY_EXIT_K for ablations. The value is baked into the ckpt's
# dat_extra_args, so SFT/eval inherit it automatically.
EE_K="${DAT_HD_EARLY_EXIT_K:-16}"
echo "[early-exit] hd_early_exit_k=${EE_K}"

# nogate by default (inherited from 0701). DAT_HD_GATE_INIT for A/B only.
HD_GATE_ARG=()
if [[ -n "${DAT_HD_GATE_INIT:-}" ]]; then
    HD_GATE_ARG=(--dat_hd_gate_init "${DAT_HD_GATE_INIT}")
    echo "[gate] enabling hd_gate_init=${DAT_HD_GATE_INIT}"
fi

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40861}" llava/train/train_qwen_dat.py \
    "${HD_GATE_ARG[@]}" \
    --dat_hd_early_exit_k "$EE_K" \
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
#   scripts/qwen2_5vl_adl_0805/exp_sft_from_nogate_ee16_frozen_mlp.sh
