#!/usr/bin/env bash

# DAT-LLaVA-1.5 GQA数据集综合测试脚本
# 基于训练脚本 train_dat_llava1_5_v2.sh 的配置
# 使用GQA数据集进行TTFT和FLOPs测试

export DS_SKIP_CUDA_CHECK=1

# 激活conda环境
source /root/miniconda3/bin/activate ml-fastvlm

# 设置实验名称和路径
EXP_NAME="tdat-7b-l0d32-s12g8z3"
OUTPUT_DIR="/data/checkpoints/weilai/$EXP_NAME"



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

# 检查GQA数据集路径
DATA_PATH="/data/gqa/testdev_balanced_questions.json"
IMAGE_FOLDER="/data/gqa/images"

if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 未找到GQA数据集文件"
    echo "请检查以下路径:"
    echo "  - $DATA_PATH"
    echo "  - $IMAGE_FOLDER"
    echo ""
    echo "请确保GQA数据集已正确下载并解压到指定位置"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "错误: 未找到GQA图像文件夹"
    echo "请检查路径: $IMAGE_FOLDER"
    exit 1
fi

echo "使用数据文件: $DATA_PATH"
echo "使用图像文件夹: $IMAGE_FOLDER"

# 创建结果目录
RESULTS_DIR="./test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "结果将保存到: $(pwd)/$RESULTS_DIR"

# 运行GQA数据集综合测试
echo "开始运行DAT-LLaVA-1.5 GQA数据集综合测试..."
echo "="*80

# 设置时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 运行综合测试脚本
echo "运行GQA数据集综合测试 (TTFT + FLOPs)"
echo "="*60
python /home/zhuofan.xia/ml-fastvlm/test_dat_llava1_5_ttft_flops.py \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --resolution 336 \
    --max-samples 1000 \
    --output-dir "$RESULTS_DIR"

TEST_SUCCESS=$?

# 汇总结果
echo ""
echo "GQA数据集测试结果汇总"
echo "="*60
echo "数据集: GQA (Graph Question Answering)"
echo "数据路径: $DATA_PATH"
echo "图像文件夹: $IMAGE_FOLDER"
echo "模型路径: $CHECKPOINT_PATH"
echo "分辨率: 336x336"
echo "视觉编码器: clip"
echo "LLM类型: llama"
echo "-"*60
if [ $TEST_SUCCESS -eq 0 ]; then
    echo "GQA综合测试: ✅ 成功"
else
    echo "GQA综合测试: ❌ 失败"
fi

# 检查测试结果
echo ""
echo "GQA数据集测试完成"
echo "结果文件保存在: $(pwd)/$RESULTS_DIR"
echo ""
echo "生成的文件:"
ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "未找到结果文件"

# 检查测试结果
if [ $TEST_SUCCESS -eq 0 ]; then
    echo ""
    echo "🎉 GQA数据集测试完成成功!"
    echo "结果文件保存在: $(pwd)/$RESULTS_DIR"
    echo ""
    echo "生成的文件:"
    ls -la "$RESULTS_DIR"/*.json 2>/dev/null || echo "未找到结果文件"
else
    echo ""
    echo "❌ GQA数据集测试失败，请检查日志"
    exit 1
fi
