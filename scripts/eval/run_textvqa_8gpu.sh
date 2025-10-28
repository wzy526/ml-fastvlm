#!/usr/bin/env bash

# 8卡并行：使用LLaVA官方流程生成 TextVQA 预测并计算分数
# 用法：bash scripts/eval/run_textvqa_8gpu.sh

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/checkpoints/weilai/tdat-7b-l0d32-s12g8z3}
QUESTION_FILE=${QUESTION_FILE:-/data/textvqa/llava_textvqa_val_v051_ocr.jsonl}
ANNOTATION_FILE=${ANNOTATION_FILE:-/data/textvqa/TextVQA_0.5.1_val.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/data/textvqa/train_images}
OUT_DIR=${OUT_DIR:-/root/ml-fastvlm/textvqa_results}
CONV_MODE=${CONV_MODE:-vicuna_v1}

mkdir -p "$OUT_DIR"

NUM_SHARDS=${NUM_SHARDS:-8}

echo "[TextVQA-8GPU] 使用LLaVA官方流程生成分片预测 NUM_SHARDS=$NUM_SHARDS"

# 检查输出目录
if [ ! -d "$OUT_DIR" ]; then
  echo "创建输出目录: $OUT_DIR"
  mkdir -p "$OUT_DIR"
fi

# 检查必要文件是否存在
if [ ! -f "$QUESTION_FILE" ]; then
  echo "错误: 问题文件不存在: $QUESTION_FILE"
  exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
  echo "错误: 图像目录不存在: $IMAGE_FOLDER"
  exit 1
fi

if [ ! -f "$ANNOTATION_FILE" ]; then
  echo "错误: 标注文件不存在: $ANNOTATION_FILE"
  exit 1
fi

echo "开始8卡并行预测（使用LLaVA model_vqa_loader）..."

# 清理之前的进程
pkill -f "model_vqa_loader" 2>/dev/null || true
sleep 2

# 启动8个GPU进程
for ((i=0; i<NUM_SHARDS; i++)); do
  echo "启动GPU $i..."
  CUDA_VISIBLE_DEVICES=$i \
  stdbuf -oL -eL python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$OUT_DIR/tdat-7b.s${i}.jsonl" \
    --temperature 0 \
    --conv-mode "$CONV_MODE" \
    --num-chunks $NUM_SHARDS \
    --chunk-idx $i \
    2>&1 | sed -u "s/^/[GPU ${i}] /" | tee "$OUT_DIR/textvqa_s${i}.log" &
  
  # 每个进程启动后等待一下，避免资源竞争
  sleep 1
done

echo "等待所有GPU完成..."
wait

echo "检查生成的文件..."
ls -la "$OUT_DIR"/tdat-7b.s*.jsonl || echo "警告: 没有找到预测文件"

echo "[TextVQA-8GPU] 合并预测..."
cat "$OUT_DIR"/tdat-7b.s*.jsonl > "$OUT_DIR"/tdat-7b.jsonl

echo "[TextVQA-8GPU] 使用LLaVA官方评估器计算TextVQA分数..."
python -m llava.eval.eval_textvqa \
  --annotation-file "$ANNOTATION_FILE" \
  --result-file "$OUT_DIR/tdat-7b.jsonl"

echo "[TextVQA-8GPU] 完成"


