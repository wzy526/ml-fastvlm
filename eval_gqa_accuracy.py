#!/usr/bin/env python3
"""
GQA准确率评估脚本
基于训练好的DAT-LLaVA-1.5模型进行GQA数据集准确率评估
"""

import os
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
import re
from collections import defaultdict

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM
import transformers


def load_model_and_tokenizer(model_path, device_map="auto"):
    """加载模型和分词器"""
    disable_torch_init()
    
    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    # 加载模型
    if 'qwen' in model_path.lower():
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map=device_map
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map=device_map
        )
    
    # 设置图像处理器
    image_processor = transformers.CLIPImageProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    
    return model, tokenizer, image_processor


def load_gqa_data(data_path, max_samples=None):
    """加载GQA数据"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 转换为列表格式
    samples = []
    for qid, item in data.items():
        samples.append({
            'questionId': qid,
            'question': item['question'],
            'imageId': item['imageId'],
            'answer': item['answer'],
            'fullAnswer': item.get('fullAnswer', item['answer'])
        })
    
    if max_samples:
        samples = samples[:max_samples]
    
    return samples


def evaluate_single_sample(model, tokenizer, image_processor, sample, image_folder, conv_mode="llava_v1"):
    """评估单个样本"""
    try:
        # 加载图像
        image_path = os.path.join(image_folder, f"{sample['imageId']}.jpg")
        if not os.path.exists(image_path):
            return None, "Image not found"
        
        image = Image.open(image_path).convert('RGB')
        
        # 构建对话
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{sample['question']}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 处理输入
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()
        
        # 处理图像
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # 生成回答
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=32,
                use_cache=True
            )
        
        # 解码输出
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs[len(prompt):].strip()
        
        return outputs, None
        
    except Exception as e:
        return None, str(e)


def normalize_answer(answer):
    """标准化答案用于比较"""
    if answer is None:
        return ""
    
    # 转换为小写
    answer = answer.lower().strip()
    
    # 移除标点符号
    answer = re.sub(r'[^\w\s]', '', answer)
    
    # 移除多余空格
    answer = ' '.join(answer.split())
    
    return answer


def calculate_accuracy(predictions, ground_truths):
    """计算准确率"""
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truths):
        if normalize_answer(pred) == normalize_answer(gt):
            correct += 1
    
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="GQA准确率评估")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--data-path", type=str, required=True, help="GQA数据文件路径")
    parser.add_argument("--image-folder", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--output-file", type=str, default="gqa_accuracy_results.json", help="输出文件")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="对话模式")
    
    args = parser.parse_args()
    
    print("="*80)
    print("GQA准确率评估")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.data_path}")
    print(f"图像文件夹: {args.image_folder}")
    print(f"最大样本数: {args.max_samples or '全部'}")
    print("="*80)
    
    # 加载模型
    print("加载模型...")
    model, tokenizer, image_processor = load_model_and_tokenizer(args.model_path)
    print("模型加载完成")
    
    # 加载数据
    print("加载GQA数据...")
    samples = load_gqa_data(args.data_path, args.max_samples)
    print(f"加载了 {len(samples)} 个样本")
    
    # 评估
    print("开始评估...")
    predictions = []
    ground_truths = []
    errors = []
    
    for i, sample in enumerate(tqdm(samples, desc="评估进度")):
        pred, error = evaluate_single_sample(
            model, tokenizer, image_processor, sample, args.image_folder, args.conv_mode
        )
        
        if error:
            errors.append(f"样本 {i}: {error}")
            predictions.append("")
        else:
            predictions.append(pred)
        
        ground_truths.append(sample['answer'])
    
    # 计算准确率
    accuracy = calculate_accuracy(predictions, ground_truths)
    
    # 保存结果
    results = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'image_folder': args.image_folder,
        'total_samples': len(samples),
        'accuracy': accuracy,
        'predictions': predictions,
        'ground_truths': ground_truths,
        'errors': errors,
        'error_count': len(errors)
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    print(f"总样本数: {len(samples)}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"错误数: {len(errors)}")
    print(f"结果已保存到: {args.output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
