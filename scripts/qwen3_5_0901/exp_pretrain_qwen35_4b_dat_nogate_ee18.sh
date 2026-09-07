#!/usr/bin/env bash
set -euo pipefail

# 0901 Stage-1 pretrain: Qwen3.5-4B DAT (nogate) + HD ViT early exit k=18.
# ============================================================================
#
# Arm 2/3 of the early-exit sweep (see exp_pretrain_qwen35_4b_dat_nogate.sh
# for the k=0 baseline, *_ee12.sh for the aggressive cut):
#   k=0  -> full depth        (24/24 blocks, baseline)
#   k=18 -> 75% depth  <-- this script  (Qwen2.5 k=24/32 equivalent)
#   k=12 -> 50% depth         (Qwen2.5 k=16/32 equivalent)
# NOTE: Qwen3.5's ViT has 24 uniform full-attention blocks (NOT Qwen2.5's 32
# window+full mix), so k=24 would be a no-op here; the qwen2.5 k values were
# mapped by depth fraction. HD ViT time scales ~k/24. Early exit only affects
# the separate HD path; LR keeps full depth. k_proj_hd/v_proj_hd bind to the
# trained feature depth -> SFT and inference MUST use the same k.
#
# Everything else identical to the k=0 baseline recipe:
#   - NO hd_gate, intention branch + intention_as_gate ON, spatial guide OFF
#   - grid_size 20, off_grps 8, inter_size 128, hr_scale 3, hd_proj True
#   - trainable: DAT modules ONLY @ lr 1e-4 (ViT/merger/LLM frozen)
#   - data: llava_sa1b_caption_pretrain.json (503k SA-1B captions)
#
# 4B-specific: 32 LLM layers, full-attn at 3,7,11,15,19,23,27,31.
# --dat_layers auto resolves to 8 DAT layers (2B had 6). Do not hardcode.
#
# After a successful run the final model is also staged to pod-local
# /workspace/model_cache/<EXP_NAME> so a same-pod SFT loads from local disk.
#
# Launch from repo root on any B200/B300 8-GPU pod that mounts this CFS:
#   cd /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm
#   bash scripts/qwen3_5_0901/exp_pretrain_qwen35_4b_dat_nogate_ee18.sh

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
CFS_4B_CANDIDATES=(
    "${CFS_4B:-}"
    "$LOCAL_CACHE/Qwen3.5-4B"
    /home/ea-cv-nlp-train-offline-2/xzf/models/Qwen3.5-4B
    "$XZF_ROOT/models/Qwen3.5-4B"
    /home/cvfa-multimodal-comprehension/fanwenxiao/data/model/Qwen3.5-4B
    /home/ea-cvfa-aigc-x2v-2/lixueheng/model/Qwen3.5-4B
)

pick_4b() {
    local p
    for p in "${CFS_4B_CANDIDATES[@]}"; do
        [[ -n "$p" && -f "$p/config.json" ]] && { echo "$p"; return 0; }
    done
    return 1
}

SRC_4B="$(pick_4b)" || {
    echo "[ERROR] Qwen3.5-4B not found. Copy it to $LOCAL_CACHE/Qwen3.5-4B" >&2
    exit 1
}

# Prefer pod-local disk for the HF load; rsync once if cache is empty.
MODEL_PATH="${MODEL_PATH:-$LOCAL_CACHE/Qwen3.5-4B}"
if [[ "$SRC_4B" != "$MODEL_PATH" ]]; then
    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        echo "[0901-pretrain-ee] rsync 4B $SRC_4B -> $MODEL_PATH"
        mkdir -p "$LOCAL_CACHE"
        rsync -a --info=progress2 "$SRC_4B/" "$MODEL_PATH/"
    fi
fi

EARLY_K="${EARLY_K:-18}"
IMAGE_ROOT="${IMAGE_ROOT:-$XZF_ROOT/sft_data}"
DATA_JSON="${DATA_JSON:-$XZF_ROOT/json_files/llava_sa1b_caption_pretrain.json}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/cache/vldat}"
EXP_NAME="${EXP_NAME:-0901_pretrain_qwen35_4b_dat_nogate_ee${EARLY_K}}"

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

# Micro-batch defaults by GPU class. B300-class (>=200GB): bsz16 x 8ranks x
# accum1 -> GLOBAL batch 128 (0902 experiment; the 0826-lineage recipe was 64).
# lr stays at the recipe value unless LR env is set (sqrt-law for 128 would be
# ~1.4x). Rollback: PER_DEVICE_BATCH=8 GRAD_ACCUM=1 (global 64, bigger kernels)
# or PER_DEVICE_BATCH=4 GRAD_ACCUM=2 (original). Small cards keep 4x2=64.
# -i 0: single-line output — a `| head -1` here dies to pipefail+SIGPIPE races
GPU_MEM_MB="$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo 0)"
if [[ "${GPU_MEM_MB%%.*}" -ge 200000 ]]; then
    DEF_BSZ=16; DEF_ACCUM=1
else
    DEF_BSZ=4; DEF_ACCUM=2
fi

DAT_LAYERS="${DAT_LAYERS:-auto}"
NPROC="${NPROC:-8}"

echo "[0901-pretrain-ee] qwen3_5 4B  dat_layers=$DAT_LAYERS  early_k=$EARLY_K  bsz=${PER_DEVICE_BATCH:-$DEF_BSZ}x${GRAD_ACCUM:-$DEF_ACCUM}  model=$MODEL_PATH  data=$(basename "$DATA_JSON")"

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
echo "[0901-pretrain-ee] staging final model $STAGE_SRC -> $STAGE_DST"
if rsync -a --exclude='checkpoint-*' --exclude='tb' "$STAGE_SRC/" "$STAGE_DST.tmp/" \
        && rm -rf "$STAGE_DST" && mv "$STAGE_DST.tmp" "$STAGE_DST"; then
    echo "[0901-pretrain-ee] staged: $STAGE_DST"
else
    echo "[WARN] staging to $STAGE_DST failed; same-pod SFT falls back to CFS" >&2
fi
