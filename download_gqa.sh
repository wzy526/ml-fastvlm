#!/bin/bash

# GQA数据集下载脚本
# 下载GQA数据集到/cluster/home/data目录

set -e  # 遇到错误时退出

# 设置路径
DATA_DIR="/perception-hl/zhuofan.xia/data"
GQA_DIR="$DATA_DIR/gqa"

# GQA数据集下载链接
GQA_QUESTIONS_URL="https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
GQA_IMAGES_URL="https://downloads.cs.stanford.edu/nlp/data/gqa/allImages.zip"
GQA_SCENE_GRAPHS_URL="https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip"

echo "============================================================"
echo "GQA数据集下载脚本"
echo "============================================================"
echo "目标目录: $GQA_DIR"
echo ""

# 创建目录
echo "创建目录..."
mkdir -p "$GQA_DIR"
cd "$GQA_DIR"

echo "当前工作目录: $(pwd)"
echo ""

# 下载GQA问题文件
echo "1. 下载GQA问题文件..."
if [ ! -f "questions1.2.zip" ]; then
    echo "下载 questions1.2.zip..."
    wget -c "$GQA_QUESTIONS_URL" -O questions1.2.zip
else
    echo "questions1.2.zip 已存在，跳过下载"
fi

# 下载GQA图像文件
echo ""
echo "2. 下载GQA图像文件..."
if [ ! -f "allImages.zip" ]; then
    echo "下载 allImages.zip (约2.5GB)..."
    wget -c "$GQA_IMAGES_URL" -O allImages.zip
else
    echo "allImages.zip 已存在，跳过下载"
fi

# 下载场景图文件（可选）
echo ""
echo "3. 下载GQA场景图文件..."
if [ ! -f "sceneGraphs.zip" ]; then
    echo "下载 sceneGraphs.zip..."
    wget -c "$GQA_SCENE_GRAPHS_URL" -O sceneGraphs.zip
else
    echo "sceneGraphs.zip 已存在，跳过下载"
fi

echo ""
echo "============================================================"
echo "开始解压文件..."
echo "============================================================"

# 解压问题文件
echo "解压问题文件..."
if [ ! -d "questions" ]; then
    unzip -q questions1.2.zip
    echo "问题文件解压完成"
else
    echo "questions目录已存在，跳过解压"
fi

# 解压图像文件
echo "解压图像文件..."
if [ ! -d "images" ]; then
    echo "解压图像文件 (这可能需要一些时间)..."
    unzip -q allImages.zip
    echo "图像文件解压完成"
else
    echo "images目录已存在，跳过解压"
fi

# 解压场景图文件
echo "解压场景图文件..."
if [ ! -d "sceneGraphs" ]; then
    unzip -q sceneGraphs.zip
    echo "场景图文件解压完成"
else
    echo "sceneGraphs目录已存在，跳过解压"
fi

echo ""
echo "============================================================"
echo "GQA数据集下载和解压完成！"
echo "============================================================"
echo "数据集位置: $GQA_DIR"
echo ""
echo "目录结构:"
echo "  $GQA_DIR/"
echo "  ├── questions/          # 问题文件"
echo "  │   ├── train_balanced_questions.json"
echo "  │   ├── val_balanced_questions.json"
echo "  │   └── ..."
echo "  ├── images/             # 图像文件"
echo "  │   ├── 000000.jpg"
echo "  │   ├── 000001.jpg"
echo "  │   └── ..."
echo "  └── sceneGraphs/        # 场景图文件"
echo "      ├── train_sceneGraphs.json"
echo "      ├── val_sceneGraphs.json"
echo "      └── ..."
echo ""

# 显示文件统计信息
echo "文件统计信息:"
if [ -d "questions" ]; then
    echo "问题文件数量: $(ls questions/*.json 2>/dev/null | wc -l)"
fi

if [ -d "images" ]; then
    echo "图像文件数量: $(ls images/*.jpg 2>/dev/null | wc -l)"
fi

if [ -d "sceneGraphs" ]; then
    echo "场景图文件数量: $(ls sceneGraphs/*.json 2>/dev/null | wc -l)"
fi

echo ""
echo "============================================================"
echo "更新ttft_test.sh中的路径配置"
echo "============================================================"

# 更新ttft_test.sh中的路径
TTFT_SCRIPT="/cluster/home2/wzy/ml-fastvlm/ttft_test.sh"
if [ -f "$TTFT_SCRIPT" ]; then
    echo "更新 $TTFT_SCRIPT 中的路径..."
    
    # 备份原文件
    cp "$TTFT_SCRIPT" "$TTFT_SCRIPT.backup"
    
    # 更新路径
    sed -i "s|DATA_PATH=\"/path/to/gqa/questions.json\"|DATA_PATH=\"$GQA_DIR/questions/val_balanced_questions.json\"|g" "$TTFT_SCRIPT"
    sed -i "s|IMAGE_FOLDER=\"/path/to/gqa/images\"|IMAGE_FOLDER=\"$GQA_DIR/images\"|g" "$TTFT_SCRIPT"
    
    echo "路径更新完成！"
    echo "  DATA_PATH: $GQA_DIR/questions/val_balanced_questions.json"
    echo "  IMAGE_FOLDER: $GQA_DIR/images"
else
    echo "警告: 未找到 $TTFT_SCRIPT"
fi

echo ""
echo "============================================================"
echo "下载完成！现在可以运行TTFT测试了"
echo "============================================================"
echo "运行命令:"
echo "  cd /cluster/home2/wzy/ml-fastvlm"
echo "  bash ttft_test.sh"
echo "" 