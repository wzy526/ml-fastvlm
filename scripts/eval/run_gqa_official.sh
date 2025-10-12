#!/usr/bin/env bash

# 基于LLaVA官方实现的GQA评估脚本
# 适配您训练的DAT-LLaVA-1.5模型

export DS_SKIP_CUDA_CHECK=1

# 激活conda环境
source /home/zhuofan.xia/miniconda3/bin/activate pt260

# 设置实验名称和路径
EXP_NAME="tdat-7b-l0d32-s12g8z3"
OUTPUT_DIR="/perception-hl/zhuofan.xia/vlm_exps/textdat/$EXP_NAME"
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

# 检查GQA数据集路径
DATA_PATH="/perception-hl/zhuofan.xia/data/gqa/val_balanced_questions.json"
IMAGE_FOLDER="/perception-hl/zhuofan.xia/data/gqa/images"

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
echo "使用模型路径: $CHECKPOINT_PATH"
echo "评估模式: LLaVA官方风格GQA评估"

# 创建结果目录
RESULTS_DIR="./gqa_official_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "结果将保存到: $(pwd)/$RESULTS_DIR"

# 运行GQA官方风格评估
echo "开始运行LLaVA官方风格GQA评估..."
echo "="*80

# 设置时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 运行评估脚本
echo "运行GQA官方风格评估"
echo "="*60
python eval_gqa_official.py \
    --model-path "$CHECKPOINT_PATH" \
    --data-path "$DATA_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --output-file "$RESULTS_DIR/gqa_results_${TIMESTAMP}.jsonl" \
    --conv-mode "vicuna_v1" \
    --temperature 0

EVAL_SUCCESS=$?

# 汇总结果
echo ""
echo "GQA官方风格评估结果汇总"
echo "="*60
echo "数据集: GQA val_balanced_questions.json"
echo "数据路径: $DATA_PATH"
echo "图像文件夹: $IMAGE_FOLDER"
echo "模型路径: $CHECKPOINT_PATH"
echo "对话模式: vicuna_v1"
echo "温度: 0"
echo "-"*60
if [ $EVAL_SUCCESS -eq 0 ]; then
    echo "GQA官方风格评估: ✅ 成功"
else
    echo "GQA官方风格评估: ❌ 失败"
fi

# 检查评估结果
echo ""
echo "GQA官方风格评估完成"
echo "结果文件保存在: $(pwd)/$RESULTS_DIR"
echo ""
echo "生成的文件:"
ls -la "$RESULTS_DIR"/*.json* 2>/dev/null || echo "未找到结果文件"

# 显示准确率结果
if [ $EVAL_SUCCESS -eq 0 ]; then
    echo ""
    echo "🎉 GQA官方风格评估完成成功!"
    echo "结果文件保存在: $(pwd)/$RESULTS_DIR"
    echo ""
    echo "生成的文件:"
    ls -la "$RESULTS_DIR"/*.json* 2>/dev/null || echo "未找到结果文件"
    
    # 显示准确率
    SUMMARY_FILE="$RESULTS_DIR/gqa_results_${TIMESTAMP}_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo ""
        echo "准确率结果:"
        python -c "
import json
with open('$SUMMARY_FILE', 'r') as f:
    data = json.load(f)
print(f'总样本数: {data[\"total_samples\"]}')
print(f'准确率: {data[\"accuracy\"]:.4f} ({data[\"accuracy\"]*100:.2f}%)')
print(f'错误数: {data[\"error_count\"]}')
print(f'对话模式: {data[\"conv_mode\"]}')
print(f'温度: {data[\"temperature\"]}')
"
    fi
else
    echo ""
    echo "❌ GQA官方风格评估失败，请检查日志"
    exit 1
fi
