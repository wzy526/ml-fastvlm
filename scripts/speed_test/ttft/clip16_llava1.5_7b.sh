#!/bin/bash

# LLaVA-1.5-7B with LLaVA-1.6 Vision Encoder TTFT 测试脚本
# Vision Encoder: CLIP-ViT-L/14-336px-LLaVA16 (CLIPVisionTowerLLaVA16)
# LLM: LLaVA-1.5-7B
# 使用LLaVA-1.6风格的分块处理高分辨率图像

echo "=========================================="
echo "LLaVA-1.5-7B with LLaVA-1.6 Vision Encoder TTFT 测试"
echo "Vision Encoder: CLIP-ViT-L/14-336px-LLaVA16"
echo "LLM: LLaVA-1.5-7B"
echo "LLaVA-1.6 style patching for high-resolution images"
echo "=========================================="

# 模型路径 - LLaVA-1.5-7B模型
MODEL_PATH="./checkpoints/llava-1.5-7b"

# 数据路径
DATA_PATH="/cluster/nvme2/wzy/gqa/questions/testdev_balanced_questions.json"
IMAGE_FOLDER="/cluster/nvme2/wzy/gqa/images"

# 测试参数
RESOLUTION=672  # LLaVA-1.6支持的高分辨率
MAX_SAMPLES="" 
VISUAL_TOKENS=2880  # LLaVA-1.6在672x672分辨率下：5个patches × 576 tokens = 2880
OUTPUT_FILE="ttft_test_results_llava1.5_7b_llava16_vision_672x672_vt${VISUAL_TOKENS}_full.json"

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请确保LLaVA-1.5-7B模型已下载并放置在正确位置"
    exit 1
fi

# 检查数据路径是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件不存在: $DATA_PATH"
    echo "请确保GQA数据文件已下载"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "错误: 图像文件夹不存在: $IMAGE_FOLDER"
    echo "请确保GQA图像文件夹已下载"
    exit 1
fi

echo "开始LLaVA-1.5-7B with LLaVA-1.6 Vision Encoder TTFT测试..."
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "图像路径: $IMAGE_FOLDER"
echo "分辨率: ${RESOLUTION}x${RESOLUTION}"
echo "最大样本数: ${MAX_SAMPLES:-全部样本}"
echo "Visual Tokens: $VISUAL_TOKENS"
echo "输出文件: $OUTPUT_FILE"
echo ""
echo "模型配置："
echo "- Vision Encoder: CLIP-ViT-L/14-336px-LLaVA16"
echo "- LLM: LLaVA-1.5-7B"
echo "- LLaVA-1.6风格分块: 使用process_anyres_image"
echo "- 网格配置: [336, 672, 1008]"
echo "- 分块大小: 336px"
echo "- 分辨率: ${RESOLUTION}x${RESOLUTION}"
echo "- Visual Tokens: $VISUAL_TOKENS (5个patches × 576 tokens)"
echo "- 特征融合: 保持空间位置信息"
echo ""

# 运行TTFT测试
python ttft_test.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --vision-encoder "clip_llava16" \
    --resolution "$RESOLUTION" \
    ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"} \
    --visual-tokens "$VISUAL_TOKENS" \
    --output-file "$OUTPUT_FILE" \
    --conv-mode "llava_v1"

# 检查测试是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "LLaVA-1.5-7B with LLaVA-1.6 Vision Encoder TTFT测试完成！"
    echo "结果保存在: $OUTPUT_FILE"
    echo "=========================================="
    
    # 显示结果摘要
    if [ -f "$OUTPUT_FILE" ]; then
        echo "结果摘要："
        echo "Vision Encoder: CLIP-ViT-L/14-336px-LLaVA16"
        echo "LLM: LLaVA-1.5-7B"
        echo "分辨率: ${RESOLUTION}x${RESOLUTION}"
        echo "分块策略: LLaVA-1.6 style patching"
        cat "$OUTPUT_FILE" | grep -E '"avg_ttft_ms"|"total_samples"|"vision_encoder"|"llm"'
    fi
else
    echo ""
    echo "=========================================="
    echo "测试失败！请检查错误信息"
    echo "可能的原因："
    echo "1. LLaVA-1.5-7B模型路径不正确"
    echo "2. 缺少LLaVA-1.6视觉编码器相关依赖"
    echo "3. 模型配置不正确"
    echo "=========================================="
    exit 1
fi 