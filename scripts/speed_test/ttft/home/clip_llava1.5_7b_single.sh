#!/bin/bash

# LLaVA-1.5-7B + CLIP 单卡TTFT 测试脚本
# Vision Encoder: CLIP-ViT-L/14-336px
# LLM: LLaVA-1.5-7B
# 使用单卡测试

echo "=========================================="
echo "LLaVA-1.5-7B + CLIP 单卡TTFT 测试"
echo "Vision Encoder: CLIP-ViT-L/14-336px"
echo "LLM: LLaVA-1.5-7B"
echo "单卡测试"
echo "=========================================="

# 模型路径 - LLaVA-1.5-7B模型
MODEL_PATH="./checkpoints/llava-1.5-7b"

# 数据路径
DATA_PATH="/cluster/nvme2/wzy/gqa/questions/testdev_balanced_questions.json"
IMAGE_FOLDER="/cluster/nvme2/wzy/gqa/images"

# 测试参数
RESOLUTION=336  # LLaVA-1.5支持的分辨率：336（fastvlm原文），672
MAX_SAMPLES=""  
VISUAL_TOKENS=576  # CLIP在336x336分辨率下的visual token数量：(336/14)^2 = 576
OUTPUT_FILE="ttft_test_results_llava1.5_7b_336x336_vt${VISUAL_TOKENS}_single.json"

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

echo "开始LLaVA-1.5-7B + CLIP 单卡TTFT测试..."
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "图像路径: $IMAGE_FOLDER"
echo "分辨率: ${RESOLUTION}x${RESOLUTION}"
echo "最大样本数: ${MAX_SAMPLES:-全部样本}"
echo "Visual Tokens: $VISUAL_TOKENS"
echo "输出文件: $OUTPUT_FILE"
echo "测试配置: 单卡"
echo ""
echo "模型配置："
echo "- Vision Encoder: CLIP-ViT-L/14-336px"
echo "- LLM: LLaVA-1.5-7B"
echo "- 分辨率: ${RESOLUTION}x${RESOLUTION}"
echo "- Visual Tokens: $VISUAL_TOKENS"
echo "- 模型大小: 7B参数"
echo "- 对话模式: LLaVA v1"
echo "- 测试模式: 单卡"
echo ""

# 设置环境变量优化内存使用
echo "设置环境变量以优化内存使用..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# 构建单卡测试命令
echo "启动单卡测试..."
CMD_ARGS=(
    python
    ttft_test.py
    --model-path "$MODEL_PATH"
    --data-path "$DATA_PATH"
    --image-folder "$IMAGE_FOLDER"
    --vision-encoder "clip"
    --resolution "$RESOLUTION"
    --visual-tokens "$VISUAL_TOKENS"
    --output-file "$OUTPUT_FILE"
    --conv-mode "llava_v1"
)

# 只有当MAX_SAMPLES不为空时才添加该参数
if [ -n "$MAX_SAMPLES" ]; then
    CMD_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

# 运行单卡测试
"${CMD_ARGS[@]}"

# 检查测试是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "LLaVA-1.5-7B + CLIP 单卡TTFT测试完成！"
    echo "结果保存在: $OUTPUT_FILE"
    echo "=========================================="
    
    # 显示结果摘要
    if [ -f "$OUTPUT_FILE" ]; then
        echo "结果摘要："
        echo "Vision Encoder: CLIP-ViT-L/14-336px"
        echo "LLM: LLaVA-1.5-7B"
        echo "分辨率: ${RESOLUTION}x${RESOLUTION}"
        echo "测试模式: 单卡"
        echo "模型: LLaVA-1.5-7B"
        cat "$OUTPUT_FILE" | grep -E '"avg_ttft_ms"|"total_samples"|"vision_encoder"|"llm"|"world_size"'
    fi
else
    echo ""
    echo "=========================================="
    echo "测试失败！请检查错误信息"
    echo "可能的原因："
    echo "1. LLaVA-1.5-7B模型路径不正确"
    echo "2. 缺少CLIP视觉编码器相关依赖"
    echo "3. 模型配置不正确"
    echo "4. GPU内存不足"
    echo "5. Python环境问题"
    echo "=========================================="
    exit 1
fi 