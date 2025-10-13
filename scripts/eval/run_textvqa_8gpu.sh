#!/usr/bin/env bash

# 8卡并行：生成 TextVQA 预测并计算分数
# 用法：bash scripts/eval/run_textvqa_8gpu.sh

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/perception-hl/zhuofan.xia/vlm_exps/textdat/tdat-7b-l0d32-s12g8z3}
QUESTION_FILE=${QUESTION_FILE:-/perception-hl/zhuofan.xia/data/textvqa/val_questions.json}
ANNOTATION_FILE=${ANNOTATION_FILE:-/perception-hl/zhuofan.xia/data/textvqa/val_annotations.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/perception-hl/zhuofan.xia/data/textvqa/train_images}
OUT_DIR=${OUT_DIR:-/ephstorage/vlm_exps/textdat}
CONV_MODE=${CONV_MODE:-llava_v1}

mkdir -p "$OUT_DIR"

NUM_SHARDS=${NUM_SHARDS:-8}

echo "[TextVQA-8GPU] 生成分片预测 NUM_SHARDS=$NUM_SHARDS"

for ((i=0; i<NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES=$i \
  python -u -m llava.eval.evaluate_textvqa \
    --model-path "$MODEL_PATH" \
    --conv-mode "$CONV_MODE" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$OUT_DIR/textvqa_val_pred.s${i}.jsonl" \
    --temperature 0 --num_beams 1 --max_new_tokens 16 \
    --num-shards $NUM_SHARDS --shard-id $i \
    > "$OUT_DIR/textvqa_s${i}.log" 2>&1 &
done

wait

echo "[TextVQA-8GPU] 合并预测..."
cat "$OUT_DIR"/textvqa_val_pred.s*.jsonl > "$OUT_DIR"/textvqa_val_pred.jsonl

echo "[TextVQA-8GPU] 计算ANLS分数..."
python -u -m llava.eval.textvqa_eval \
  --annotation-file "$ANNOTATION_FILE" \
  --result-file "$OUT_DIR"/textvqa_val_pred.jsonl

echo "[TextVQA-8GPU] 完成"


