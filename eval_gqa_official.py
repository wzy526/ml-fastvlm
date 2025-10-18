#!/usr/bin/env python3
"""
基于LLaVA官方实现的GQA评估脚本
适配您训练的DAT-LLaVA-1.5模型
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
    """加载模型和分词器 - 基于官方实现"""
    disable_torch_init()
    
    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    # 修复配置中的 decoder_config 问题
    config = transformers.AutoConfig.from_pretrained(model_path)
    if hasattr(config, 'decoder_config') and isinstance(config.decoder_config, dict):
        print("修复 decoder_config 配置...")
        if 'model_type' in config.decoder_config:
            decoder_config = transformers.AutoConfig.from_dict(config.decoder_config)
            config.decoder_config = decoder_config
        else:
            from transformers import PretrainedConfig
            decoder_config = PretrainedConfig.from_dict(config.decoder_config)
            config.decoder_config = decoder_config
    
    # 加载模型
    model_name = get_model_name_from_path(model_path)
    if 'qwen' in model_name.lower():
        print("Loading LlavaQwen2ForCausalLM model (Qwen2 backbone)")
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=device_map
        )
    else:
        print("Loading LlavaLlamaForCausalLM model (Llama backbone)")
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=device_map
        )
    
    # 修复视觉编码器加载问题
    print("初始化视觉编码器...")
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            print("加载视觉编码器...")
            vision_tower.load_model()
            print("视觉编码器加载完成")
    
    # 设置图像处理器
    try:
        image_processor = transformers.CLIPImageProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
    except OSError as e:
        if "preprocessor_config.json" in str(e):
            print("警告: 未找到preprocessor_config.json，使用训练时的视觉编码器...")
            vision_tower_path = "/home/zhuofan.xia/gsva_pretrains/clip-vit-large-patch14-336"
            try:
                image_processor = transformers.CLIPImageProcessor.from_pretrained(
                    vision_tower_path, trust_remote_code=True
                )
                print(f"成功加载视觉编码器: {vision_tower_path}")
            except Exception as vision_error:
                print(f"视觉编码器加载失败: {vision_error}")
                # 创建基本图像处理器
                from transformers import CLIPImageProcessor
                image_processor = CLIPImageProcessor(
                    size={"height": 336, "width": 336},
                    do_convert_rgb=True,
                    do_normalize=True,
                    do_rescale=True,
                    do_resize=True,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711]
                )
                print("使用基本图像处理器配置")
        else:
            raise e
    
    return model, tokenizer, image_processor


def load_gqa_data(data_path, max_samples=None):
    """加载GQA数据 - 转换为LLaVA格式"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 转换为LLaVA格式的样本列表
    samples = []
    for qid, item in data.items():
        sample = {
            'questionId': qid,
            'question': item['question'],
            'imageId': item['imageId'],
            'answer': item['answer'],
            'fullAnswer': item.get('fullAnswer', item['answer'])
        }
        samples.append(sample)
    
    if max_samples:
        samples = samples[:max_samples]
    
    return samples


def evaluate_single_sample(model, tokenizer, image_processor, sample, image_folder, conv_mode="llava_v1", temperature=0):
    """评估单个样本 - 修复推理问题"""
    try:
        # 加载图像 - 修复图像路径匹配问题
        image_id = sample['imageId']
        
        # 尝试不同的文件名格式 - 根据测试结果，优先尝试n前缀
        possible_paths = [
            os.path.join(image_folder, f"n{image_id}.jpg"),  # 优先尝试n前缀
            os.path.join(image_folder, f"{image_id}.jpg"),
            os.path.join(image_folder, f"n{image_id}.png"),
            os.path.join(image_folder, f"{image_id}.png")
        ]
        
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            return None, f"Image not found for ID {image_id}. Tried: {possible_paths[:2]}"
        
        image = Image.open(image_path).convert('RGB')
        
        # 构建对话 - 使用llava_v1格式（与训练时一致）
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{sample['question']}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 处理输入
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(model.device)
        
        # 处理图像
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # 生成回答 - 修复推理参数
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(model.device),
                do_sample=False,  # 固定为False，确保确定性
                temperature=0,   # 固定为0
                top_p=None,
                num_beams=1,
                max_new_tokens=32,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id  # 添加pad_token_id
            )
        
        # 解码输出 - 修复解码逻辑
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # 提取回答部分 - 更安全的提取方式
        if prompt in outputs:
            outputs = outputs.split(prompt)[-1].strip()
        else:
            # 如果找不到prompt，尝试其他方式
            outputs = outputs.strip()
        
        # 清理输出
        if outputs:
            # 移除可能的重复内容
            lines = outputs.split('\n')
            if lines:
                outputs = lines[0].strip()
        
        return outputs, None
        
    except Exception as e:
        print(f"推理错误: {e}")
        return None, str(e)


def normalize_answer(answer):
    """标准化答案 - 与GQA官方评估一致"""
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


def save_results_jsonl(predictions, ground_truths, samples, output_file):
    """保存为JSONL格式 - 兼容GQA官方评估"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (pred, gt, sample) in enumerate(zip(predictions, ground_truths, samples)):
            result = {
                'questionId': sample['questionId'],
                'question': sample['question'],
                'imageId': sample['imageId'],
                'ground_truth': gt,
                'prediction': pred,
                'correct': normalize_answer(pred) == normalize_answer(gt)
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="GQA官方风格评估")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--data-path", type=str, required=True, help="GQA数据文件路径")
    parser.add_argument("--image-folder", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--output-file", type=str, default="gqa_results.jsonl", help="输出文件")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="对话模式")
    parser.add_argument("--temperature", type=float, default=0, help="生成温度")
    parser.add_argument("--chunks", type=int, default=1, help="分块数（兼容官方接口）")
    parser.add_argument("--chunk-idx", type=int, default=0, help="当前块索引（兼容官方接口）")
    
    args = parser.parse_args()
    
    print("="*80)
    print("GQA官方风格评估")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.data_path}")
    print(f"图像文件夹: {args.image_folder}")
    print(f"最大样本数: {args.max_samples or '完整测试集'}")
    print(f"对话模式: {args.conv_mode}")
    print(f"温度: {args.temperature}")
    print("="*80)
    
    # 加载模型
    print("加载模型...")
    model, tokenizer, image_processor = load_model_and_tokenizer(args.model_path)
    print("模型加载完成")
    
    # 加载数据
    print("加载GQA数据...")
    samples = load_gqa_data(args.data_path, args.max_samples)
    print(f"加载了 {len(samples)} 个样本")
    
    # 分块处理（兼容官方接口）
    if args.chunks > 1:
        chunk_size = len(samples) // args.chunks
        start_idx = args.chunk_idx * chunk_size
        end_idx = start_idx + chunk_size if args.chunk_idx < args.chunks - 1 else len(samples)
        samples = samples[start_idx:end_idx]
        print(f"处理块 {args.chunk_idx}/{args.chunks-1}: 样本 {start_idx}-{end_idx-1}")
    
    # 评估
    print("开始评估...")
    predictions = []
    ground_truths = []
    errors = []
    
    for i, sample in enumerate(tqdm(samples, desc="评估进度")):
        pred, error = evaluate_single_sample(
            model, tokenizer, image_processor, sample, args.image_folder, 
            args.conv_mode, args.temperature
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
    save_results_jsonl(predictions, ground_truths, samples, args.output_file)
    
    # 保存汇总结果
    summary_file = args.output_file.replace('.jsonl', '_summary.json')
    summary_results = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'image_folder': args.image_folder,
        'total_samples': len(samples),
        'accuracy': accuracy,
        'error_count': len(errors),
        'conv_mode': args.conv_mode,
        'temperature': args.temperature,
        'chunks': args.chunks,
        'chunk_idx': args.chunk_idx
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    print(f"总样本数: {len(samples)}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"错误数: {len(errors)}")
    print(f"结果文件: {args.output_file}")
    print(f"汇总文件: {summary_file}")
    print("="*80)


if __name__ == "__main__":
    main()
