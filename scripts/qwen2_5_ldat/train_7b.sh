#!/usr/bin/env bash
# Native Qwen2.5-VL-7B -> optional answer SFT -> where-head -> learned DAT.
# Run in the existing training environment. DRY_RUN=1 prints exact commands.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUT="${OUT:-$HOME/vldat_experiments/qwen25_7b_ldat}"
DATA_ROOT="${DATA_ROOT:-$HOME/models_data}"
DATA_JSON="${DATA_JSON:-$DATA_ROOT/extra_0916/viscot_bbox.train.json}"
DAT_JSON="${DAT_JSON:-$DATA_JSON}"
REF_JSON="${REF_JSON:-$DATA_ROOT/extra_1018/refcoco_bbox.json}"
COCO_JSON="${COCO_JSON:-$DATA_ROOT/extra_1018/coco_small_bbox.json}"
IMAGE_ROOT="${IMAGE_ROOT:-$HOME/sft_data/train_split}"
STAGES="${STAGES:-where dat}"
NPROC="${NPROC:-8}"
PORT="${PORT:-29681}"
LR_TOK="${LR_TOK:-640}"
WHERE_LAYER="${WHERE_LAYER:-8}"
WHERE_PT="${WHERE_PT:-$OUT/where_lm${WHERE_LAYER}.pt}"
LDAT_PT="${LDAT_PT:-$OUT/ldat.pt}"
MERGED_DIR="${MERGED_DIR:-$OUT/sft-merged}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "${N_WIN:-1}" != 1 && "${GRID:-0}" == 0 ]]; then
    echo 'Set GRID explicitly for multi-window training (e.g. GRID=20).' >&2; exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

run() {
    printf '%q ' "$@"; printf '\n'
    if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}
need_file() {
    if [[ "$DRY_RUN" != 1 && ! -f "$1" ]]; then echo "Missing file: $1" >&2; exit 1; fi
}
if [[ "$DRY_RUN" != 1 ]]; then mkdir -p "$OUT"; fi

for stage in $STAGES; do
    case "$stage" in
    sft)
        need_file "$DATA_JSON"
        run torchrun --nproc_per_node "$NPROC" --master_port "$PORT" llava/train/train_qwen_dat.py \
            --model_family qwen2_5_vl --model_name_or_path "$MODEL_PATH" --use_dat False \
            --data_path "$DATA_JSON" --image_folder "$IMAGE_ROOT" \
            --lr_min_pixels 784 --lr_max_pixels "$((LR_TOK * 784))" \
            --lora_enable True --lora_r 8 --lora_alpha 16 --lora_target_layers qkvo \
            --lora_lr "${LORA_LR:-2e-5}" --tune_mm_vision False --tune_mm_mlp True \
            --tune_mm_llm False --mm_projector_lr 5e-6 --kd_on False \
            --deepspeed scripts/zero_configs/zero2.json --bf16 True --tf32 True \
            --output_dir "$OUT/sft" --num_train_epochs 1 \
            --per_device_train_batch_size "${PER_DEVICE_BATCH:-1}" \
            --gradient_accumulation_steps "${SFT_ACCUM:-8}" --eval_strategy no \
            --save_strategy epoch --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0 \
            --warmup_steps 20 --lr_scheduler_type cosine --logging_steps 1 \
            --model_max_length 4096 --gradient_checkpointing True \
            --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
            --dataloader_num_workers "${DATALOADER_WORKERS:-4}" \
            --seed "${SEED:-42}" --report_to "${REPORT_TO:-none}"
        run python scripts/merge_native_lora.py --model_path "$MODEL_PATH" \
            --lora_path "$OUT/sft" --output_dir "$MERGED_DIR"
        MODEL_PATH="$MERGED_DIR"
        ;;
    where)
        need_file "$DATA_JSON"; need_file "$REF_JSON"; need_file "$COCO_JSON"
        run torchrun --nproc_per_node "$NPROC" --master_port "$((PORT + 1))" scripts/where_head.py \
            --model_family qwen2_5_vl --model_path "$MODEL_PATH" \
            --train_json "$DATA_JSON" "$REF_JSON" "$COCO_JSON" --image_root "$IMAGE_ROOT" \
            --max_per_json 40000 40000 60000 --heldout "${WHERE_HELDOUT:-200}" \
            --head xattn --d 384 --xlayers 2 --xheads 6 --feat "lm:$WHERE_LAYER" \
            --head_prefix train --lr_tok "$LR_TOK" --ce_weight 20 --dilate 0 \
            --steps "${WHERE_STEPS:-1500}" --accum "${WHERE_ACCUM:-4}" --lr 1e-4 \
            --eval_every "${WHERE_EVAL_EVERY:-200}" --vstar_n 0 --stagger 0 \
            --seed "${SEED:-42}" --save "$WHERE_PT"
        ;;
    dat)
        need_file "$DAT_JSON"; need_file "$WHERE_PT"
        run torchrun --nproc_per_node "$NPROC" --master_port "$((PORT + 2))" scripts/offset_feat_train.py \
            --model_family qwen2_5_vl --model_path "$MODEL_PATH" \
            --where_head_pt "$WHERE_PT" --where_layer "$WHERE_LAYER" \
            --train_json "$DAT_JSON" --image_root "$IMAGE_ROOT" \
            --held_from_train "${DAT_HELDOUT:-2000}" --held_eval_n "${HELD_EVAL_N:-200}" \
            --max_train "${MAX_TRAIN:-60000}" --train_cfg ldat --cfgs crop,dat,ldat \
            --ldat_in native --ldat_win dat --pe global --ldat_d 256 \
            --lr_tok "$LR_TOK" --crop_tok "${CROP_TOK:-400}" --n_win "${N_WIN:-1}" --grid "${GRID:-0}" \
            --dat_alpha 0.8 --dat_ratio 0 --dat_k 0.6 --src_scale 4 --supersample 2 \
            --hr_cap "${HR_CAP:-4096}" --ans_w 1 --box_w "${BOX_W:-0.1}" \
            --hinge_w "${HINGE_W:-0}" --tr_w "${TR_W:-0.1}" \
            --lr "${DAT_LR:-1e-4}" --steps "${DAT_STEPS:-2000}" --accum "${DAT_ACCUM:-1}" \
            --eval_every "${DAT_EVAL_EVERY:-250}" --save_best --seed "${SEED:-42}" --save "$LDAT_PT"
        ;;
    *) echo "Unknown stage: $stage (sft, where, dat)" >&2; exit 1 ;;
    esac
done
echo "Backbone: $MODEL_PATH"
echo "Where head: $WHERE_PT"
echo "Density head: $LDAT_PT"
