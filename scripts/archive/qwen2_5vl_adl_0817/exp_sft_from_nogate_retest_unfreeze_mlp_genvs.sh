#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"

# 0817 Stage-2 SFT: 0731 nogate pretrain (0701 architecture, retest run)
# + the 0817 "general + visual search" mix (v2 of the 0812 mix).
# ============================================================================
#
# Identical architecture/hparams to exp_sft_from_nogate_retest_unfreeze_mlp_
# ocrvs.sh (0812) — the ONLY variable is the data:
#
#   0812: llava_hr_ocr_vs_0812.json   478k  (OCR/VS-augmented; verdict: OCR
#         flat, needle -1.6 from anchor dilution, general mixed)
#   0817: llava_hr_gen_vs_0817.json  ~517k
#         needle anchors UP  : sa1b x1.5 (75k), visualprobe x2 (11.5k),
#                              deepeyes kept (24.3k) -> anchor share 16.7%
#         proven-dead OCR cut: synthdog 0, ocr_vqa 30k, stvqa 5k,
#                              hd251k chartqa/textvqa/ai2d removed
#         general attack     : densefusion 100k (GPT-4V dense captions,
#                              high-res), allava 50k (GPT-4V instruct QA),
#                              aokvqa ~17k + scienceqa ~6.3k (MC letters)
#
#   Built by construct_sft_0817.py. Goal: hold vstar/HRBench at the 0731
#   level (>=76.0 / >=65.5 / >=62.0) while pushing MMBench/SEED/MME/SQA
#   toward base Qwen2.5-VL. See chat 0817.
#
# Architecture (MUST match the 0731 pretrain ckpt — 0701 family):
# - NO hd_gate, NO hd_skip_merger_mlp, full HD ViT depth, grid_size 20
# - LSE merge / STE / intention_branch + intention_as_gate ON
# - LLM LoRA r=8 a=16 lr=2e-5; projector tuned lr=5e-6; DAT lr=1e-4; LR ViT frozen

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
MODEL_PATH="${MODEL_PATH:-$OSS_EXP/0731_pretrain_nogate_retest}"
CACHE_ROOT="${CACHE_ROOT:-$LOCAL_ROOT/cache/vldat}"
EXP_NAME="${EXP_NAME:-0817_sft_from_nogate_retest_unfreeze_mlp_genvs}"

# train_split is a SYMLINK FARM on a LOCAL fs (OSS FUSE can't create symlinks).
IMAGE_ROOT="${IMAGE_ROOT:-$LOCAL_ROOT/sft_data}"

DATA_JSON="${DATA_JSON:-$OSS_DATA/llava_hr_gen_vs_0817.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing $DATA_JSON" >&2
    echo "        Build it first via construct_sft_0817.py (download + merge phases)." >&2
    exit 1
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
    echo "[ERROR] Missing pretrain ckpt: $MODEL_PATH" >&2
    echo "        (the 0731/0701-architecture nogate pretrain, archived on OSS)" >&2
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

# HD gate: MUST match the pretrain ckpt's architecture (nogate by default).
HD_GATE_ARG=()
if [[ -n "${DAT_HD_GATE_INIT:-}" ]]; then
    HD_GATE_ARG=(--dat_hd_gate_init "${DAT_HD_GATE_INIT}")
    echo "[gate] enabling hd_gate_init=${DAT_HD_GATE_INIT}"
fi

# NOTE: intentionally no --dat_hd_skip_merger_mlp / --dat_hd_early_exit_k
# flags here -- this pretrain ckpt predates both features (full-depth HD ViT
# through the merger MLP). Passing them would build adapters of the wrong
# width (5120 vs 2048) and crash on ckpt load.

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40975}" llava/train/train_qwen_dat.py \
    "${HD_GATE_ARG[@]}" \
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
