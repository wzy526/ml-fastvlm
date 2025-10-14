#!/usr/bin/env python3
"""
基于LLaVA官方repo的OCR tokens生成脚本
使用EasyOCR提取TextVQA图像中的文本
"""

import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

def extract_ocr_with_easyocr(image_path, reader):
    """使用EasyOCR提取文本 - LLaVA官方方法"""
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return ""
        
        # 执行OCR
        results = reader.readtext(image_path)
        
        # 提取文本和置信度
        texts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # 只保留高置信度的文本
                texts.append(text.strip())
        
        return " ".join(texts)
    except Exception as e:
        print(f"OCR处理错误 {image_path}: {e}")
        return ""

def generate_ocr_tokens_llava_style(question_file, image_folder, output_file, max_samples=None):
    """按照LLaVA官方方式生成OCR tokens"""
    print("初始化EasyOCR...")
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)  # 使用GPU加速
        print("EasyOCR初始化完成")
    except ImportError:
        print("错误: EasyOCR未安装，请安装: pip install easyocr")
        return
    except Exception as e:
        print(f"EasyOCR初始化失败: {e}")
        return
    
    # 加载问题数据
    print(f"加载问题文件: {question_file}")
    with open(question_file, 'r') as f:
        question_data = json.load(f)
    
    questions = question_data['questions']
    if max_samples:
        questions = questions[:max_samples]
    
    print(f"处理 {len(questions)} 个问题...")
    
    # 按image_id分组，避免重复处理同一张图像
    image_ocr_cache = {}
    ocr_results = []
    
    for i, q in enumerate(tqdm(questions)):
        question_id = q['question_id']
        image_id = q['image_id']
        
        # 检查是否已经处理过这张图像
        if image_id in image_ocr_cache:
            ocr_text = image_ocr_cache[image_id]
        else:
            # 尝试不同的图像路径
            possible_paths = [
                os.path.join(image_folder, f"{image_id}.jpg"),
                os.path.join(image_folder, f"{image_id}.png"),
                os.path.join(image_folder, f"n{image_id}.jpg"),
                os.path.join(image_folder, f"n{image_id}.png")
            ]
            
            image_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            if not image_path:
                print(f"警告: 找不到图像 {image_id}")
                ocr_text = ""
            else:
                # 提取OCR文本
                ocr_text = extract_ocr_with_easyocr(image_path, reader)
            
            # 缓存结果
            image_ocr_cache[image_id] = ocr_text
        
        # 保存结果 - 按照LLaVA官方格式
        ocr_results.append({
            "question_id": question_id,
            "ocr_tokens": ocr_text
        })
        
        # 每100个样本保存一次中间结果
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个样本，缓存了 {len(image_ocr_cache)} 张图像")
            # 保存中间结果
            temp_file = output_file + ".temp"
            with open(temp_file, 'w') as f:
                json.dump(ocr_results, f, indent=2)
    
    # 保存最终结果
    print(f"保存OCR tokens到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(ocr_results, f, indent=2)
    
    # 删除临时文件
    temp_file = output_file + ".temp"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"完成! 生成了 {len(ocr_results)} 个OCR tokens")
    print(f"处理了 {len(image_ocr_cache)} 张唯一图像")

def main():
    parser = argparse.ArgumentParser(description="按照LLaVA官方方式生成TextVQA OCR tokens")
    parser.add_argument("--question-file", default="/perception-hl/zhuofan.xia/data/textvqa/val_questions.json")
    parser.add_argument("--image-folder", default="/perception-hl/zhuofan.xia/data/textvqa/train_images")
    parser.add_argument("--output-file", default="/perception-hl/zhuofan.xia/data/textvqa/val_ocr_tokens.json")
    parser.add_argument("--max-samples", type=int, default=None, help="处理样本数量（None表示处理全部）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.question_file):
        print(f"错误: 问题文件不存在: {args.question_file}")
        return
    
    if not os.path.exists(args.image_folder):
        print(f"错误: 图像目录不存在: {args.image_folder}")
        return
    
    generate_ocr_tokens_llava_style(args.question_file, args.image_folder, args.output_file, args.max_samples)

if __name__ == "__main__":
    main()
