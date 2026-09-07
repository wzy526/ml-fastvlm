#!/usr/bin/env bash
set -euo pipefail

# 0901 Stage-2 SFT: Qwen3.5-4B DAT (nogate, HD early exit k=12) + 369k ivcap mix.
# ============================================================================
#
# Consumes stage-1 ckpt from exp_pretrain_qwen35_4b_dat_nogate_ee12.sh
# (resolved pod-local /workspace/model_cache first, then CFS with
# auto-staging to local; explicit MODEL_PATH env overrides both).
# DAT args MUST match stage 1 — including dat_hd_early_exit_k: the
# k_proj_hd/v_proj_hd adapters bind to the block-12 feature depth. Inference
# must also run with hd_early_exit_k=12 (that is the whole point: HD ViT
# time ~50% of full depth; see *_ee18.sh for the 75% arm and
# exp_sft_qwen35_4b_dat_ivcap.sh for the k=0 baseline).
#
# Trainable set matches the k=0 baseline SFT:
#   LLM LoRA r=8 a=16 lr=2e-5; visual.merger lr=5e-6; DAT lr=1e-4; ViT frozen.
#
# Data is the mix that is actually on this CFS:
#   llava_hr_essential_sa1b_ivcap.json (369k)
#
# Launch from repo root:
#   cd /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm
#   bash scripts/qwen3_5_0901/exp_sft_qwen35_4b_dat_ivcap_ee12.sh

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# torchrun puts the script dir (llava/train), not CWD, on sys.path — the
# repo root must be exported explicitly on fresh pods (no editable install).
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Silence known-noisy third-party warnings (cutlass CuTe-DSL deprecation spam
# during FA4 JIT, swig shutdown spam, torch's pynvml notice). Inherited by
# torchrun ranks; override by presetting PYTHONWARNINGS.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::DeprecationWarning:cutlass,ignore:builtin type:DeprecationWarning,ignore:The pynvml package is deprecated:FutureWarning}"
export NUMEXPR_MAX_THREADS=4 NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
EARLY_K="${EARLY_K:-12}"
LOCAL_CACHE="${LOCAL_CACHE:-/workspace/model_cache}"
STAGE1_NAME="0901_pretrain_qwen35_4b_dat_nogate_ee${EARLY_K}"

# Resolve stage-1 ckpt: explicit MODEL_PATH env > pod-local cache > CFS.
# The CFS branch stages the model to local disk first (tmp+mv, so a
# half-copied dir never passes the config.json probe on a later run) and
# falls back to loading straight from CFS if the copy fails.
if [[ -z "${MODEL_PATH:-}" ]]; then
    LOCAL_S1="$LOCAL_CACHE/$STAGE1_NAME"
    CFS_S1="$CKPT_ROOT/$STAGE1_NAME"
    if [[ -f "$LOCAL_S1/config.json" ]]; then
        MODEL_PATH="$LOCAL_S1"
    elif [[ -f "$CFS_S1/config.json" ]]; then
        echo "[0901-sft-ee] staging stage-1 $CFS_S1 -> $LOCAL_S1"
        if rsync -a --exclude='checkpoint-*' --exclude='tb' "$CFS_S1/" "$LOCAL_S1.tmp/" \
                && rm -rf "$LOCAL_S1" && mv "$LOCAL_S1.tmp" "$LOCAL_S1"; then
            MODEL_PATH="$LOCAL_S1"
        else
            echo "[WARN] staging to $LOCAL_S1 failed; loading stage-1 from CFS" >&2
            MODEL_PATH="$CFS_S1"
        fi
    else
        MODEL_PATH="$CFS_S1"
    fi
fi
echo "[0901-sft-ee] stage-1 model: $MODEL_PATH"
IMAGE_ROOT="${IMAGE_ROOT:-$XZF_ROOT/sft_data}"
DATA_JSON="${DATA_JSON:-$XZF_ROOT/json_files/llava_hr_essential_sa1b_ivcap.json}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/cache/vldat}"
EXP_NAME="${EXP_NAME:-0901_sft_qwen35_4b_dat_ivcap_ee${EARLY_K}}"

# External training platform tails tensorboard events under this pod-local
# path; fall back to CFS if this pod doesn't have it.
TB_ROOT="${TB_ROOT:-/export/App/training_platform/PinoModel/models}"
TB_DIR="$TB_ROOT/$EXP_NAME"
if ! mkdir -p "$TB_DIR" 2>/dev/null; then
    TB_DIR="$CKPT_ROOT/$EXP_NAME/tb"
    echo "[WARN] $TB_ROOT not writable; tensorboard events -> $TB_DIR"
    mkdir -p "$TB_DIR"
fi
# transformers >= 5.10 ignores --logging_dir; the TB callback reads this env.
export TENSORBOARD_LOGGING_DIR="$TB_DIR"
echo "[tb] events -> $TB_DIR"

if [[ ! -f "$DATA_JSON" ]]; then echo "[ERROR] Missing $DATA_JSON" >&2; exit 1; fi
if [[ ! -d "$IMAGE_ROOT/train_split" ]]; then
    echo "[ERROR] Missing $IMAGE_ROOT/train_split" >&2; exit 1
fi
for prefix in sa1b coco vg gqa ocr_vqa docvqa infovqa synthdog; do
    if [[ ! -e "$IMAGE_ROOT/train_split/$prefix" ]]; then
        echo "[ERROR] Missing $IMAGE_ROOT/train_split/$prefix" >&2
        exit 1
    fi
done
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[ERROR] Missing stage-1 ckpt: $MODEL_PATH" >&2
    echo "        Run exp_pretrain_qwen35_4b_dat_nogate_ee12.sh first." >&2
    exit 1
fi

python - <<'PY'
import sys
import transformers
print(f"[preflight] transformers {transformers.__version__}")
try:
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
except ImportError:
    sys.exit("[preflight ERROR] need transformers >= 5.10 with Qwen3_5ForConditionalGeneration")
import flash_attn
print(f"[preflight] flash_attn {flash_attn.__version__}")
import llava  # requires PYTHONPATH=repo root (exported above); fails fast here
print("[preflight] llava import OK")
try:
    import fla
    print(f"[preflight] fla {getattr(fla, '__version__', '?')}")
except ImportError:
    print("[preflight WARN] fla missing — GDN falls back to slow torch path")
PY

mkdir -p "$CKPT_ROOT/$EXP_NAME"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

# Reduce fragmentation-driven OOM at large micro-batches (no numerics impact).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG=WARN  # platform injects INFO by default — too chatty
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-7200}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"

# Micro-batch defaults by GPU class. B300-class (>=200GB): bsz16 x 8ranks x
# accum1 -> GLOBAL batch 128 (0902 request, matching pretrain; measured
# headroom is huge — pretrain bsz8 peaked ~25GB/275GB). Small cards keep
# 4x2=64. Override via PER_DEVICE_BATCH / GRAD_ACCUM env.
# -i 0: single-line output — a `| head -1` here dies to pipefail+SIGPIPE races
GPU_MEM_MB="$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)"
if [[ "${GPU_MEM_MB%%.*}" -ge 200000 ]]; then
    DEF_BSZ=16; DEF_ACCUM=1
else
    DEF_BSZ=4; DEF_ACCUM=2
fi

DAT_LAYERS="${DAT_LAYERS:-auto}"
NPROC="${NPROC:-8}"

echo "[0901-sft-ee] qwen3_5 4B  dat_layers=$DAT_LAYERS  early_k=$EARLY_K  bsz=${PER_DEVICE_BATCH:-$DEF_BSZ}x${GRAD_ACCUM:-$DEF_ACCUM}  model=$MODEL_PATH  data=$(basename "$DATA_JSON")"

torchrun --nproc_per_node="$NPROC" --master_port "${MASTER_PORT:-40989}" llava/train/train_qwen_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen3_5 \
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
    --dat_hd_early_exit_k "$EARLY_K" \
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
    --ddp_timeout 7200 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH:-$DEF_BSZ}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-$DEF_ACCUM}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-500}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --learning_rate "${LR:-2e-5}" \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS:-50}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length "${MODEL_MAX_LEN:-262144}" \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --group_by_modality_length True \
    --dataloader_num_workers "${DATALOADER_WORKERS:-12}" \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --seed 42 \
    --report_to tensorboard \
    --run_name "$EXP_NAME"

source "$(dirname "${BASH_SOURCE[0]}")/../qwen2_5vl_adl_0701/_merge_after_train.sh"
