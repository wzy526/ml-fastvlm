#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"

# 0812 Stage-2 SFT: on top of the 0731 nogate pretrain (0701 architecture,
# retest run) + the 0812 OCR/visual-search-augmented SFT mix.
# ============================================================================
#
# DIFFERENT ARCHITECTURE FAMILY from the 0805 k32_hdskipmlp line — do not
# confuse the two pretrain ckpts:
#
#   0731_pretrain_nogate_retest (this script's source):
#     - NO hd_skip_merger_mlp: HD features go THROUGH the merger MLP (2048-dim)
#     - projector was TRAINED during pretrain (tune_mm_mlp=True, lr=1e-4) and
#       continues drifting during SFT (tune_mm_mlp=True, lr=5e-6)
#     - HD ViT early-exit didn't exist yet at 0701/0731 -> always full depth
#       (equivalent to k=0, but the flag was never introduced for this line
#       so it must stay OMITTED, not set to 0 -- omitting vs. passing
#       --dat_hd_early_exit_k 0 are numerically identical, but the CLI arg
#       simply doesn't apply here; keep parity with the original 0701 script)
#
#   0805 k32_hdskipmlp line (see exp_sft_from_nogate_k32_hdskipmlp_ocrvs.sh):
#     - hd_skip_merger_mlp=True: HD features bypass the MLP (5120-dim)
#     - projector FROZEN everywhere
#
# Passing hd_skip_merger_mlp/mismatched mm_projector settings against the
# WRONG pretrain ckpt produces a shape mismatch (2048 vs 5120) or silently
# wrong finetuning of a frozen-at-pretrain-time projector.
#
# The ONLY variable vs the original exp_sft_from_nogate_unfreeze_mlp.sh
# (0701) run is the SFT data:
#
#   original (0731 retest): llava_hr_essential_sa1b_ivcap.json      369k
#   0812:                   llava_hr_ocr_vs_0812.json              ~478k
#          = 369k base
#          + chartqa/textvqa/ai2d from llava_hd251k                53k  (ai2d
#            is the lmms-lab TEST split -- fine since ai2d isn't in our eval
#            suite, but drop it via --no_ai2d in construct_sft_0812.py if
#            that ever changes)
#          + ST-VQA (scene text)                                   26k
#          + DeepEyes-47k visual_toolbox subsample                 24k  (V*-
#            style visual search, verifiable short answers; RL tool-call
#            instructions stripped)
#          + Mini-o3 VisualProbe train                             5.7k (hardest
#            visual search: small targets + many distractors)
#
#   Built by construct_sft_0812.py. See that script + chat 0812 (advisor:
#   DeepEyes / Mini-o3 / vision-OPD line) for the full rationale.
#
# Architecture (MUST match the pretrain ckpt):
# - LSE merge / STE / no D1 / no D3 / no F1 / no dirA
# - intention_branch + intention_as_gate ON
# - NO hd_gate, NO hd_skip_merger_mlp, full HD ViT depth
# - DAT_LAYERS = DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL
#
# Trainable set
# -------------
# - LLM         LoRA r=8 / alpha=16 / target_layers=all / lr=2e-5
# - projector   tune_mm_mlp=True, mm_projector_lr=5e-6
# - DAT         all DAT params trainable (NO hd_gate), lr=1e-4
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
# Source = the 0731 nogate pretrain retest ckpt, archived on OSS.
MODEL_PATH="${MODEL_PATH:-$OSS_EXP/0731_pretrain_nogate_retest}"
CACHE_ROOT="${CACHE_ROOT:-$LOCAL_ROOT/cache/vldat}"
EXP_NAME="${EXP_NAME:-0812_sft_from_nogate_retest_unfreeze_mlp_ocrvs}"

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

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40974}" llava/train/train_qwen_dat.py \
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
