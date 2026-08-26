#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"

# 0826 Stage-1 pretrain: Qwen3.5-2B DAT (nogate), OSS cluster.
# ============================================================================
#
# First DAT run on the hybrid Qwen3.5 family (f43bcdf port). Kept as close as
# possible to the proven qwen2.5 0701/0731 recipe:
#   - NO hd_gate, intention branch + intention_as_gate ON, spatial guide OFF
#   - grid_size 20, off_grps 8, inter_size 128, hr_scale 3, hd_proj True
#   - trainable set: DAT modules ONLY @ lr 1e-4 (ViT/merger/LLM frozen)
#   - data: llava_sa1b_caption_pretrain.json (503k SA-1B captions)
#
# Family-specific differences (do NOT "fix" these back):
#   - dat_layers "auto": D layers must sit on full_attention slots of the
#     hybrid stack (2B: layers 3,7,11,15,19,23 → 6 DAT layers, same count as
#     the qwen2.5 1-in-6 pattern). Resolved inside convert_qwen3_5_to_dat.
#   - deepspeed zero2 + non-reentrant GC: the configuration the port was
#     verified with on the dev pod.
#   - patch factor is 32 (not 28); handled at runtime by the processor.
#   - NO hd_early_exit_k / hd_skip_merger_mlp: those are qwen2.5 HD-ViT
#     branch tricks and do not exist in this port.
#
# Requirements (checked by the preflight below):
#   - transformers >= 5.10 (Qwen3.5 hybrid classes; the old lock's 5.6.1 is
#     NOT enough — upgrade the env or use a dedicated one)
#   - flash_attn 2.x (DAT LSE path), fla / flash-linear-attention (GDN kernels)

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config (OSS cluster) --------
OSS_DATA="${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}"
OSS_CKPT="${OSS_CKPT:-/data/oss_bucket_0/wangziyi/official_ckpt}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/pingping.wzy}"

DATA_ROOT="${DATA_ROOT:-$OSS_DATA/sft_data}"
MODEL_PATH="${MODEL_PATH:-$OSS_CKPT/Qwen3.5-2B}"
CKPT_ROOT="${CKPT_ROOT:-$LOCAL_ROOT/vldat_experiments}"
CACHE_ROOT="${CACHE_ROOT:-$LOCAL_ROOT/cache/vldat}"
EXP_NAME="${EXP_NAME:-0826_pretrain_qwen35_2b_dat_nogate}"

# train_split is a SYMLINK FARM on a LOCAL fs (OSS FUSE can't create symlinks).
IMAGE_ROOT="${IMAGE_ROOT:-$LOCAL_ROOT/sft_data}"

DATA_JSON="${DATA_JSON:-$DATA_ROOT/llava_sa1b_caption_pretrain.json}"

if [[ ! -f "$DATA_JSON" ]]; then echo "[ERROR] Missing data file: $DATA_JSON" >&2; exit 1; fi
if [[ ! -d "$IMAGE_ROOT/train_split" ]]; then echo "[ERROR] Missing $IMAGE_ROOT/train_split (create it on LOCAL disk; OSS can't hold symlinks)" >&2; exit 1; fi
if [[ ! -e "$IMAGE_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink: ln -sfn $OSS_DATA/sa1b_images $IMAGE_ROOT/train_split/sa1b" >&2; exit 1; fi
if [[ ! -d "$MODEL_PATH" ]]; then echo "[ERROR] Missing base ckpt: $MODEL_PATH" >&2; exit 1; fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then echo "[ERROR] $MODEL_PATH lacks config.json" >&2; exit 1; fi

# -------- Preflight: env must actually support Qwen3.5 --------
python - <<'PY'
import sys
import transformers
print(f"[preflight] transformers {transformers.__version__}")
try:
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
except ImportError:
    sys.exit("[preflight ERROR] transformers has no Qwen3_5ForConditionalGeneration.\n"
             "  Qwen3.5 needs transformers >= 5.10 (requirements-lock.txt's 5.6.1 predates it).\n"
             "  Upgrade in a dedicated env: pip install -U 'transformers>=5.10'")
import flash_attn
print(f"[preflight] flash_attn {flash_attn.__version__}")
try:
    import fla
    print(f"[preflight] fla {getattr(fla, '__version__', '?')}")
except ImportError:
    print("[preflight WARN] fla (flash-linear-attention) missing — GatedDeltaNet layers\n"
          "  fall back to the slow torch path. Training works but is much slower:\n"
          "  pip install flash-linear-attention")
PY

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

# fla triton autotune on step 1 can stall one rank for >10 min while the
# others wait in allreduce; default NCCL watchdog/heartbeat SIGABRTs the job.
# Raise both (paired with --ddp_timeout below).
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

# 'auto' = every full_attention slot gets a DAT layer (6 on the 2B).
DAT_LAYERS="${DAT_LAYERS:-auto}"

echo "[0826-pretrain] qwen3_5 2B  dat_layers=$DAT_LAYERS  grid=20  nogate"

torchrun --nproc_per_node=8 --master_port "${MASTER_PORT:-40979}" llava/train/train_qwen_dat.py \
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
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
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
#   scripts/qwen3_5_adl_0826/exp_sft_qwen35_2b_dat_genvs.sh
