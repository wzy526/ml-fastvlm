#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"

# 0812 Stage-2 SFT: k=32 (NO HD early exit) hdskipmlp pretrain + the 0812
# OCR/visual-search-augmented SFT mix.
# ============================================================================
#
# Architecture is UNCHANGED from exp_sft_from_nogate_k32_hdskipmlp.sh (0805):
# same k=32/no-early-exit hdskipmlp pretrain ckpt, same trainable set. The
# ONLY variable in this experiment is the SFT data:
#
#   0805:  llava_hr_essential_sa1b_ivcap.json          369k
#   0812:  llava_hr_ocr_vs_0812.json                  ~478k
#          = 369k base
#          + chartqa/textvqa/ai2d from llava_hd251k    53k  (ai2d is the
#                                                             lmms-lab TEST
#                                                             split — fine
#                                                             since ai2d is
#                                                             not in our eval
#                                                             suite, but drop
#                                                             it if that ever
#                                                             changes)
#          + ST-VQA (scene text)                       26k
#          + DeepEyes-47k visual_toolbox subsample      24k  (V*-style visual
#                                                             search, verifiable
#                                                             short answers;
#                                                             RL tool-call
#                                                             instructions
#                                                             stripped)
#          + Mini-o3 VisualProbe train                 5.7k  (hardest visual
#                                                             search: small
#                                                             targets + many
#                                                             distractors)
#
#   Built by construct_sft_0812.py. See that script + chat 0812 (advisor:
#   DeepEyes / Mini-o3 / vision-OPD line) for the full rationale.
#
# Architecture (MUST match the pretrain ckpt — dat_extra_args is built from
# CLI args, NOT read back from the ckpt config):
# - LSE merge / STE / no D1 / no dirA / NO hd_gate
# - intention_branch + intention_as_gate ON
# - hd_skip_merger_mlp = True, hd_early_exit_k = 0 (full ViT depth)
# - DAT_LAYERS = DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL
#
# Trainable set
# -------------
# - LLM         LoRA r=8 / alpha=16 / target_layers=all / lr=2e-5
# - projector   FROZEN
# - DAT         all DAT params trainable, lr=1e-4
# - LR ViT      frozen

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config (new cluster) --------
OSS_DATA="${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}"
OSS_EXP="${OSS_EXP:-/data/oss_bucket_0/wangziyi/vldat_experiments}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/pingping.wzy}"

CKPT_ROOT="${CKPT_ROOT:-$LOCAL_ROOT/vldat_experiments}"
# Source = the k=32 (no early exit) skip-mlp pretrain ckpt, archived on OSS.
MODEL_PATH="${MODEL_PATH:-$OSS_EXP/0805_pretrain_nogate_k32_hdskipmlp}"
CACHE_ROOT="${CACHE_ROOT:-$LOCAL_ROOT/cache/vldat}"
EXP_NAME="${EXP_NAME:-0812_sft_from_nogate_k32_hdskipmlp_ocrvs}"

# train_split is a SYMLINK FARM on a LOCAL fs (OSS FUSE can't create symlinks).
IMAGE_ROOT="${IMAGE_ROOT:-$LOCAL_ROOT/sft_data}"

# construct_sft_0812.py writes directly under $OSS_DATA (its SFT_DIR), not
# $OSS_DATA/sft_data.
DATA_JSON="${DATA_JSON:-$OSS_DATA/llava_hr_ocr_vs_0812.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing $DATA_JSON" >&2
    echo "        Build it first via construct_sft_0812.py (download + merge phases)." >&2
    exit 1
fi
if [[ ! -d "$IMAGE_ROOT/train_split" ]]; then echo "[ERROR] Missing $IMAGE_ROOT/train_split (create it on LOCAL disk; OSS can't hold symlinks)" >&2; exit 1; fi
if [[ ! -e "$IMAGE_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink: ln -sfn $OSS_DATA/sa1b_images $IMAGE_ROOT/train_split/sa1b" >&2; exit 1; fi
# New/extra prefixes referenced by the 0812 mix — all must be symlinked from
# the OSS train_split into the local symlink farm before training.
for prefix in chartqa textvqa ai2d stvqa deepeyes visualprobe; do
    if [[ ! -e "$IMAGE_ROOT/train_split/$prefix" ]]; then
        echo "[ERROR] Missing $prefix symlink: ln -sfn $OSS_DATA/train_split/$prefix $IMAGE_ROOT/train_split/$prefix" >&2
        exit 1
    fi
done
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing pretrain ckpt: $MODEL_PATH" >&2
    echo "        (the k=32/no-early-exit hdskipmlp pretrain, archived on OSS)" >&2
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

# nogate: MUST match the pretrain ckpt's architecture.
HD_GATE_ARG=()
if [[ -n "${DAT_HD_GATE_INIT:-}" ]]; then
    HD_GATE_ARG=(--dat_hd_gate_init "${DAT_HD_GATE_INIT}")
    echo "[gate] enabling hd_gate_init=${DAT_HD_GATE_INIT}"
fi

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40972}" llava/train/train_qwen_dat.py \
    "${HD_GATE_ARG[@]}" \
    --dat_hd_skip_merger_mlp True \
    --dat_hd_early_exit_k 0 \
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

# Auto-merge LoRA + non-LoRA trainables (DAT params) into a self-contained
# HF ckpt at $CKPT_ROOT/$EXP_NAME-merged.
source "$(dirname "${BASH_SOURCE[0]}")/../qwen2_5vl_adl_0701/_merge_after_train.sh"
