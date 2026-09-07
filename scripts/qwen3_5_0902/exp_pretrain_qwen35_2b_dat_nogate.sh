#!/usr/bin/env bash
set -euo pipefail

# 0902 Stage-1 pretrain: Qwen3.5-2B DAT (nogate), experiment pod (no internet).
# ============================================================================
#
# Small-base control arm for the 0901/0902 4B early-exit sweep. Recipe is a
# 1:1 copy of exp_pretrain_qwen35_4b_dat_nogate.sh except for the base model:
#   - NO hd_gate, intention branch + intention_as_gate ON, spatial guide OFF
#   - grid_size 20 (20x20 sampling), off_grps 8, inter_size 128, hr_scale 3
#   - hd_early_exit_k 0 -> NO HD ViT early exit (full-depth HD tower)
#   - trainable: DAT modules ONLY @ lr 1e-4 (ViT/merger/LLM frozen)
#   - data: llava_sa1b_caption_pretrain.json (503k SA-1B captions)
#
# 2B-specific: 24 layers, full-attn at 3,7,11,15,19,23 -> --dat_layers auto
# resolves to 6 DAT layers (4B had 8). Do not hardcode.
#
# After a successful run the final model is also staged to pod-local
# /workspace/model_cache/<EXP_NAME> so a same-pod SFT loads from local disk.
#
# Quick smoke run: MAX_STEPS=50 SAVE_STEPS=50 bash <this script>
#
# Launch from repo root on any B200/B300 8-GPU pod that mounts this CFS:
#   cd /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm
#   bash scripts/qwen3_5_0902/exp_pretrain_qwen35_2b_dat_nogate.sh

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
LOCAL_CACHE="${LOCAL_CACHE:-/workspace/model_cache}"
CFS_2B_CANDIDATES=(
    "${CFS_2B:-}"
    "$LOCAL_CACHE/Qwen3.5-2B"
    /home/ea-cv-nlp-train-offline-2/xzf/models/Qwen3.5-2B
    "$XZF_ROOT/models/Qwen3.5-2B"
    /home/cvfa-multimodal-comprehension/fanwenxiao/data/model/Qwen3.5-2B
    /home/ea-cvfa-aigc-x2v-2/lixueheng/model/Qwen3.5-2B
)

pick_2b() {
    local p
    for p in "${CFS_2B_CANDIDATES[@]}"; do
        [[ -n "$p" && -f "$p/config.json" ]] && { echo "$p"; return 0; }
    done
    return 1
}

SRC_2B="$(pick_2b)" || {
    echo "[ERROR] Qwen3.5-2B not found. Copy it to $LOCAL_CACHE/Qwen3.5-2B" >&2
    exit 1
}

# Prefer pod-local disk for the HF load; rsync once if cache is empty.
MODEL_PATH="${MODEL_PATH:-$LOCAL_CACHE/Qwen3.5-2B}"
if [[ "$SRC_2B" != "$MODEL_PATH" ]]; then
    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        echo "[0902-pretrain] rsync 2B $SRC_2B -> $MODEL_PATH"
        mkdir -p "$LOCAL_CACHE"
        rsync -a --info=progress2 "$SRC_2B/" "$MODEL_PATH/"
    fi
fi

IMAGE_ROOT="${IMAGE_ROOT:-$XZF_ROOT/sft_data}"
DATA_JSON="${DATA_JSON:-$XZF_ROOT/json_files/llava_sa1b_caption_pretrain.json}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/cache/vldat}"
EARLY_K="${EARLY_K:-0}"
EE_TAG=""
if [[ "$EARLY_K" != "0" ]]; then EE_TAG="_ee${EARLY_K}"; fi   # k=0 keeps the 0902 name
EXP_NAME="${EXP_NAME:-0902_pretrain_qwen35_2b_dat_nogate${EE_TAG}${EXP_SUFFIX:-}}"

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
if [[ ! -d "$IMAGE_ROOT/train_split/sa1b" ]]; then
    echo "[ERROR] Missing $IMAGE_ROOT/train_split/sa1b" >&2; exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[ERROR] Missing base ckpt: $MODEL_PATH" >&2; exit 1
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

# Micro-batch defaults by GPU class, kept identical to the 4B arms so the
# GLOBAL batch (128 on B300-class) stays comparable across bases.
# -i 0: single-line output — a `| head -1` here dies to pipefail+SIGPIPE races
GPU_MEM_MB="$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)"
if [[ "${GPU_MEM_MB%%.*}" -ge 200000 ]]; then
    DEF_BSZ=16; DEF_ACCUM=1
else
    DEF_BSZ=4; DEF_ACCUM=2
fi

DAT_LAYERS="${DAT_LAYERS:-auto}"
NPROC="${NPROC:-8}"

echo "[0902-pretrain] qwen3_5 2B  dat_layers=$DAT_LAYERS  grid=20  early_k=0  bsz=${PER_DEVICE_BATCH:-$DEF_BSZ}x${GRAD_ACCUM:-$DEF_ACCUM}  model=$MODEL_PATH  data=$(basename "$DATA_JSON")"

torchrun --nproc_per_node="$NPROC" --master_port "${MASTER_PORT:-40983}" llava/train/train_qwen_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen3_5 \
    --data_path "$DATA_JSON" \
    --image_folder "$IMAGE_ROOT/train_split" \
    --use_hr_first_resize False \
    --hd_max_pixels 5017600 \
    --use_dat True \
    --dat_layers "$DAT_LAYERS" \
    --dat_grid_size "${GRID_SIZE:-20}" \
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
    --dat_off_range "${OFF_RANGE:-0}" \
    --dat_off_penalty "${OFF_PENALTY:-0}" \
    --dat_hd_early_exit_k "$EARLY_K" \
    --dat_lr 1e-4 \
    --lora_enable False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --kd_on False \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --ddp_timeout 7200 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --max_steps "${MAX_STEPS:--1}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH:-$DEF_BSZ}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-$DEF_ACCUM}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-1000}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --learning_rate "${LR:-1e-4}" \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS:-100}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length "${MODEL_MAX_LEN:-262144}" \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --group_by_modality_length False \
    --dataloader_num_workers "${DATALOADER_WORKERS:-12}" \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --seed 42 \
    --report_to tensorboard \
    --run_name "$EXP_NAME"

# Stage the final model on pod-local disk so a same-pod SFT loads fast
# (cross-pod SFT pulls from CFS on its own). tmp+mv keeps the cache dir
# all-or-nothing: a half-copied dir must never pass the SFT config.json
# probe. Non-fatal on failure — SFT falls back to CFS.
STAGE_SRC="$CKPT_ROOT/$EXP_NAME"
STAGE_DST="$LOCAL_CACHE/$EXP_NAME"
echo "[0902-pretrain] staging final model $STAGE_SRC -> $STAGE_DST"
if rsync -a --exclude='checkpoint-*' --exclude='tb' "$STAGE_SRC/" "$STAGE_DST.tmp/" \
        && rm -rf "$STAGE_DST" && mv "$STAGE_DST.tmp" "$STAGE_DST"; then
    echo "[0902-pretrain] staged: $STAGE_DST"
else
    echo "[WARN] staging to $STAGE_DST failed; same-pod SFT falls back to CFS" >&2
fi
