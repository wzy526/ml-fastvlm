#!/usr/bin/env python3
"""
LLaVA风格的OCR tokens生成 - 测试版本
用于快速生成小批量OCR tokens进行测试
"""

import os
import json
import argparse
from tqdm import tqdm

def generate_test_ocr_tokens(question_file, image_folder, output_file, max_samples=50):
    """生成测试用的OCR tokens"""
    print("生成测试用的OCR tokens...")
    
    # 加载问题数据
    print(f"加载问题文件: {question_file}")
    with open(question_file, 'r') as f:
        question_data = json.load(f)
    
    questions = question_data['questions'][:max_samples]
    print(f"处理 {len(questions)} 个问题...")
    
    # 模拟OCR tokens（用于测试）
    ocr_results = []
    for i, q in enumerate(tqdm(questions)):
        question_id = q['question_id']
        image_id = q['image_id']
        
        # 根据问题内容生成模拟的OCR tokens
        question = q['question'].lower()
        
        # 简单的OCR tokens模拟
        if 'sign' in question or 'text' in question:
            ocr_text = "STOP SIGN"
        elif 'number' in question or 'digit' in question:
            ocr_text = "123"
        elif 'word' in question or 'letter' in question:
            ocr_text = "HELLO WORLD"
        else:
            ocr_text = "SAMPLE TEXT"
        
        ocr_results.append({
            "question_id": question_id,
            "ocr_tokens": ocr_text
        })
    
    # 保存结果
    print(f"保存OCR tokens到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(ocr_results, f, indent=2)
    
    print(f"完成! 生成了 {len(ocr_results)} 个测试OCR tokens")

def main():
    parser = argparse.ArgumentParser(description="生成测试用的OCR tokens")
    parser.add_argument("--question-file", default="/perception-hl/zhuofan.xia/data/textvqa/val_questions.json")
    parser.add_argument("--image-folder", default="/perception-hl/zhuofan.xia/data/textvqa/train_images")
    parser.add_argument("--output-file", default="/perception-hl/zhuofan.xia/data/textvqa/test_ocr_tokens.json")
    parser.add_argument("--max-samples", type=int, default=50, help="处理样本数量")
    
    args = parser.parse_args()
    
    generate_test_ocr_tokens(args.question_file, args.image_folder, args.output_file, args.max_samples)

if __name__ == "__main__":
    main()
