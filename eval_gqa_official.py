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


def load_model_and_tokenizer(model_path, device: str = "cuda"):
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
    
    # 修复 LlavaConfig 缺少必要属性的问题
    missing_attrs = {
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
        'attention_probs_dropout_prob': 0.0,
        'attention_bias': False,
        'mlp_bias': False,
        'use_cache': True,
        'rope_theta': 10000.0,
        'rope_scaling': None,
        'max_position_embeddings': 2048,
        'rms_norm_eps': 1e-6,
        'initializer_range': 0.02,
        'use_sliding_window': False,
        'sliding_window': None,
        'max_window_layers': None,
        'tie_word_embeddings': False
    }
    
    print("修复 LlavaConfig 缺失属性...")
    for attr, default_value in missing_attrs.items():
        if not hasattr(config, attr):
            setattr(config, attr, default_value)
            print(f"  添加 {attr} = {default_value}")
    
    # 加载模型
    model_name = get_model_name_from_path(model_path)
    if 'qwen' in model_name.lower():
        print("Loading LlavaQwen2ForCausalLM model (Qwen2 backbone)")
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        print("Loading LlavaLlamaForCausalLM model (Llama backbone)")
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    # 将整个模型放到指定设备，避免模块被放在 CPU 上
    model.to(device)
    model.eval()
    # 初始化视觉编码器
    print("初始化视觉编码器...")
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            print("加载视觉编码器...")
            vision_tower.load_model()
            print("视觉编码器加载完成")
        # 确保视觉编码器也在同一设备和精度
        try:
            if hasattr(vision_tower, 'to'):
                vision_tower.to(device, dtype=torch.float16)
        except Exception as _:
            pass
    # 确保多模态投影层在同一设备
    if hasattr(model, 'mm_projector') and model.mm_projector is not None:
        try:
            model.mm_projector.to(device, dtype=torch.float16)
        except Exception as _:
            pass
    
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


def evaluate_single_sample(model, tokenizer, image_processor, sample, image_folder, conv_mode="llava_v1", temperature=0, max_new_tokens=8):
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
        input_ids = input_ids.unsqueeze(0).to('cuda')
        
        # 处理图像
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # 生成回答 - 使用LLaVA官方生成参数
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to('cuda', dtype=torch.float16),
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True
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


def extract_answer_from_response(response):
    """从模型响应中提取答案 - 完全按照LLaVA官方GQA评估逻辑"""
    if not response:
        return ""
    
    # LLaVA官方GQA评估逻辑：非常简单
    # 1. 移除句末句号
    response = response.rstrip('.')
    
    # 2. 转换为小写
    response = response.lower().strip()
    
    # 3. 直接返回处理后的文本（不进行复杂的答案提取）
    return response


def normalize_answer(answer):
    """标准化答案 - 完全按照LLaVA官方GQA评估逻辑"""
    if answer is None:
        return ""
    
    # LLaVA官方GQA评估逻辑：非常简单
    # 1. 移除句末句号
    answer = answer.rstrip('.')
    
    # 2. 转换为小写
    answer = answer.lower().strip()
    
    # 3. 直接返回（不进行复杂的标准化处理）
    return answer


def gqa_accuracy(predictions, ground_truths):
    """GQA准确率计算 - 完全按照LLaVA官方评估逻辑"""
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truths):
        # 标准化预测和真实答案
        norm_pred = normalize_answer(pred)
        norm_gt = normalize_answer(gt)
        
        # LLaVA官方匹配逻辑：简单的字符串匹配
        if norm_pred == norm_gt:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def calculate_accuracy(predictions, ground_truths):
    """计算准确率"""
    correct = 0
    total = len(predictions)
    
    for pred, gt in zip(predictions, ground_truths):
        if normalize_answer(pred) == normalize_answer(gt):
            correct += 1
    
    return correct / total if total > 0 else 0.0


def save_results_jsonl(predictions, ground_truths, samples, output_file, detailed=True):
    """保存为JSONL格式 - 支持详细和LLaVA官方格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (pred, gt, sample) in enumerate(zip(predictions, ground_truths, samples)):
            if detailed:
                # 详细格式：包含所有信息用于调试和分析
                result = {
                    'questionId': sample['questionId'],
                    'question': sample['question'],
                    'imageId': sample['imageId'],
                    'ground_truth': gt,
                    'prediction': pred,
                    'prediction_normalized': normalize_answer(pred),
                    'ground_truth_normalized': normalize_answer(gt),
                    'correct': normalize_answer(pred) == normalize_answer(gt)
                }
            else:
                # LLaVA官方格式：只包含question_id和text
                result = {
                    'question_id': sample['questionId'],
                    'text': pred
                }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def convert_gqa_for_eval(src_file, dst_file):
    """转换LLaVA格式为GQA官方评估格式 - 完全按照LLaVA官方脚本"""
    all_answers = []
    for line_idx, line in enumerate(open(src_file)):
        res = json.loads(line)
        question_id = res['question_id']
        text = res['text'].rstrip('.').lower()
        all_answers.append({"questionId": question_id, "prediction": text})

    with open(dst_file, 'w') as f:
        json.dump(all_answers, f)
    
    print(f"转换完成: {src_file} -> {dst_file}")
    print(f"转换了 {len(all_answers)} 个答案")


def main():
    parser = argparse.ArgumentParser(description="GQA官方风格评估")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--data-path", type=str, required=True, help="GQA数据文件路径")
    parser.add_argument("--image-folder", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--output-file", type=str, default="gqa_results.jsonl", help="输出文件")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="对话模式")
    parser.add_argument("--temperature", type=float, default=0, help="生成温度")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="最大生成长度（建议8-16）")
    parser.add_argument("--chunks", type=int, default=1, help="分块数（兼容官方接口）")
    parser.add_argument("--chunk-idx", type=int, default=0, help="当前块索引（兼容官方接口）")
    parser.add_argument("--convert-for-eval", action="store_true", help="转换输出为GQA官方评估格式")
    parser.add_argument("--eval-file", type=str, default=None, help="GQA官方评估文件路径")
    parser.add_argument("--output-format", type=str, default="detailed", choices=["detailed", "llava"], help="输出格式：detailed(详细)或llava(官方格式)")
    
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
    model, tokenizer, image_processor = load_model_and_tokenizer(args.model_path, device="cuda")
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
            args.conv_mode, args.temperature, args.max_new_tokens
        )
        
        if error:
            errors.append(f"样本 {i}: {error}")
            predictions.append("")
        else:
            predictions.append(pred)
        
        ground_truths.append(sample['answer'])
    
    # 计算准确率 - 使用GQA官方评估逻辑
    accuracy = gqa_accuracy(predictions, ground_truths)
    
    # 保存结果
    detailed_format = (args.output_format == "detailed")
    save_results_jsonl(predictions, ground_truths, samples, args.output_file, detailed=detailed_format)
    
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
    
    # 如果指定了转换，则进行格式转换
    if args.convert_for_eval:
        if args.eval_file is None:
            eval_file = args.output_file.replace('.jsonl', '_for_eval.json')
        else:
            eval_file = args.eval_file
        
        print(f"\n转换输出为GQA官方评估格式...")
        convert_gqa_for_eval(args.output_file, eval_file)
        print(f"GQA官方评估文件: {eval_file}")
        print("="*80)


if __name__ == "__main__":
    main()
