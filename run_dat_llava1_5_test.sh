#!/usr/bin/env bash

# DAT-LLaVA-1.5 综合测试脚本
# 基于训练脚本 train_dat_llava1_5_v2.sh 的配置

# 设置环境变量
export TRANSFORMERS_OFFLINE=1
export DS_SKIP_CUDA_CHECK=1

# 激活conda环境
source /home/zhuofan.xia/miniconda3/bin/activate pt260

# 设置实验名称和路径
EXP_NAME="tdat-7b-l0d32-s12g8z3"
# 主要训练输出目录 (在另一个remote上)
OUTPUT_DIR="/perception-hl/zhuofan.xia/vlm_exps/textdat/$EXP_NAME"
# 本地备份目录 (如果存在)
BACKUP_DIR="/root/vlm_exps/textdat/$EXP_NAME"

# 检查checkpoint是否存在
if [ -d "$OUTPUT_DIR" ]; then
    echo "找到训练输出目录: $OUTPUT_DIR"
    CHECKPOINT_PATH="$OUTPUT_DIR"
elif [ -d "$BACKUP_DIR" ]; then
    echo "找到备份目录: $BACKUP_DIR"
    CHECKPOINT_PATH="$BACKUP_DIR"
else
    echo "错误: 未找到训练输出目录"
    echo "请检查以下路径:"
    echo "  - $OUTPUT_DIR"
    echo "  - $BACKUP_DIR"
    exit 1
fi

# 检查数据路径
DATA_PATH="/perception-hl/zhuofan.xia/data/llava_v1_5_mix665k.json"
IMAGE_FOLDER="/perception-hl/zhuofan.xia/data"

if [ ! -f "$DATA_PATH" ]; then
    echo "警告: 数据文件不存在: $DATA_PATH"
    echo "将使用GQA数据集进行测试"
    DATA_PATH="/root/gqa_opendatalab/testdev_balanced_questions.json"
    IMAGE_FOLDER="/root/gqa_opendatalab/images"
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 未找到测试数据文件"
    echo "请检查以下路径:"
    echo "  - /perception-hl/zhuofan.xia/data/llava_v1_5_mix665k.json"
    echo "  - /root/gqa_opendatalab/testdev_balanced_questions.json"
    exit 1
fi

echo "使用数据文件: $DATA_PATH"
echo "使用图像文件夹: $IMAGE_FOLDER"

# 创建结果目录
RESULTS_DIR="./test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"

echo "结果将保存到: $(pwd)"

# 运行综合测试
echo "开始运行DAT-LLaVA-1.5综合测试..."
echo "="*80

python /home/zhuofan.xia/ml-fastvlm/test_dat_llava1_5_ttft_flops.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --resolution 336 \
    --max-samples 1000 \
    --output-dir "$(pwd)"

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 测试完成成功!"
    echo "结果文件保存在: $(pwd)"
    echo ""
    echo "生成的文件:"
    ls -la *.json
else
    echo ""
    echo "❌ 测试失败，请检查日志"
    exit 1
fi
