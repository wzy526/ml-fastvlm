#!/usr/bin/env bash
# 在实验 pod 上把常用底座模型从 fanwenxiao 的 model 库拷到本地盘 /workspace/model_cache。
# rsync 支持断点续传, 中断后重跑即可。/workspace 是 pod 本地盘, pod 重建后需重跑。
set -euo pipefail

SRC="${SRC:-/home/cvfa-multimodal-comprehension/fanwenxiao/data/model}"
DST="${DST:-/workspace/model_cache}"
mkdir -p "$DST"

MODELS=(
    Qwen2.5-VL-3B-Instruct
    Qwen2.5-VL-7B-Instruct
    Qwen3-VL-2B-Instruct
    Qwen3-VL-8B-Instruct
    Qwen3.5-2B
    Qwen3.5-9B
)

for m in "${MODELS[@]}"; do
    echo "==== $m ===="
    rsync -a --info=progress2 "$SRC/$m" "$DST/"
done

du -sh "$DST"/*
