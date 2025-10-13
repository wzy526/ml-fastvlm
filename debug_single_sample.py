#!/usr/bin/env python3
"""
基于eval_gqa_official.py的单个样本debug脚本
"""

import os
import json
import torch
from PIL import Image
import re

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
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        print("Loading LlavaLlamaForCausalLM model (LLaMA backbone)")
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    # 初始化视觉编码器
    print("初始化视觉编码器...")
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            print("加载视觉编码器...")
            vision_tower.load_model()
            print("视觉编码器加载完成")
    
    # 加载图像处理器
    try:
        image_processor = transformers.CLIPImageProcessor.from_pretrained(model_path)
        print("使用模型路径的图像处理器")
    except OSError:
        print("模型路径没有图像处理器，使用vision_tower路径")
        vision_tower_path = "/home/zhuofan.xia/gsva_pretrains/clip-vit-large-patch14-336"
        image_processor = transformers.CLIPImageProcessor.from_pretrained(vision_tower_path)
    
    return model, tokenizer, image_processor

def evaluate_single_sample(model, tokenizer, image_processor, sample, image_folder, conv_mode="llava_v1", temperature=0):
    """评估单个样本 - 基于官方实现"""
    try:
        # 加载图像
        image_id = sample['imageId']
        
        # 尝试不同的文件名格式
        possible_paths = [
            os.path.join(image_folder, f"n{image_id}.jpg"),
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
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # 提取回答
        if prompt in outputs:
            outputs = outputs.split(prompt)[-1].strip()
        else:
            outputs = outputs.strip()
        
        if outputs:
            lines = outputs.split('\n')
            if lines:
                outputs = lines[0].strip()
        
        return outputs, None
        
    except Exception as e:
        print(f"推理错误: {e}")
        return None, str(e)

def debug_single_sample():
    """调试单个样本"""
    print("开始调试单个样本...")
    
    # 使用与eval_gqa_official.py相同的参数
    model_path = "/perception-hl/vlm_exps/textdat/tdat-7b-l0d32-s12g8z3"
    image_folder = "/perception-hl/zhuofan.xia/data/gqa/images"
    data_path = "/perception-hl/zhuofan.xia/data/gqa/val_balanced_questions.json"
    conv_mode = "llava_v1"
    
    # 加载模型
    print("加载模型...")
    model, tokenizer, image_processor = load_model_and_tokenizer(model_path)
    
    # 加载GQA数据
    print("加载GQA数据...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 获取第一个样本
    first_qid = list(data.keys())[0]
    sample = data[first_qid]
    
    print(f"测试样本:")
    print(f"  Question ID: {first_qid}")
    print(f"  Image ID: {sample['imageId']}")
    print(f"  Question: {sample['question']}")
    print(f"  Answer: {sample['answer']}")
    
    # 评估单个样本
    print("开始推理...")
    prediction, error = evaluate_single_sample(
        model, tokenizer, image_processor, sample, image_folder, conv_mode
    )
    
    if error:
        print(f"错误: {error}")
    else:
        print(f"预测结果: '{prediction}'")
        print(f"正确答案: '{sample['answer']}'")
        
        # 简单的匹配检查
        pred_clean = prediction.lower().strip() if prediction else ""
        gt_clean = sample['answer'].lower().strip()
        match = pred_clean == gt_clean
        print(f"是否匹配: {match}")
        
        # 显示完整的prompt和输出
        print("\n" + "="*50)
        print("完整推理过程:")
        
        # 重新构建对话以显示prompt
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{sample['question']}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        print(f"Prompt:\n{prompt}")
        print("="*50)

if __name__ == "__main__":
    debug_single_sample()
