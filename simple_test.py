#!/usr/bin/env python3
"""
简单的GQA测试脚本
用于诊断模型推理问题
"""

import os
import json
import torch
from PIL import Image

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM
import transformers


def test_simple_inference():
    """简单推理测试"""
    disable_torch_init()
    
    # 模型路径
    model_path = "/perception-hl/zhuofan.xia/vlm_exps/textdat/tdat-7b-l0d32-s12g8z3"
    
    print("="*60)
    print("简单GQA推理测试")
    print("="*60)
    
    # 1. 加载tokenizer
    print("1. 加载tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    print("✅ Tokenizer加载成功")
    
    # 2. 加载模型
    print("2. 加载模型...")
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
    
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ 模型加载成功")
    
    # 3. 初始化视觉编码器
    print("3. 初始化视觉编码器...")
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not vision_tower.is_loaded:
            print("加载视觉编码器...")
            vision_tower.load_model()
            print("✅ 视觉编码器加载完成")
        else:
            print("✅ 视觉编码器已加载")
    else:
        print("❌ 模型没有视觉编码器")
        return
    
    # 4. 设置图像处理器
    print("4. 设置图像处理器...")
    try:
        image_processor = transformers.CLIPImageProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        print("✅ 图像处理器加载成功")
    except OSError as e:
        if "preprocessor_config.json" in str(e):
            print("使用训练时的视觉编码器...")
            vision_tower_path = "/home/zhuofan.xia/gsva_pretrains/clip-vit-large-patch14-336"
            image_processor = transformers.CLIPImageProcessor.from_pretrained(
                vision_tower_path, trust_remote_code=True
            )
            print("✅ 使用训练时的视觉编码器")
        else:
            print(f"❌ 图像处理器加载失败: {e}")
            return
    
    # 5. 测试简单推理
    print("5. 测试简单推理...")
    
    # 使用一个简单的文本问题（不涉及图像）
    test_question = "What is 2+2?"
    
    # 构建对话
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], test_question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(f"Prompt: {prompt}")
    
    # 处理输入
    input_ids = tokenizer(prompt, return_tensors='pt')
    input_ids = input_ids['input_ids'].cuda()
    
    print(f"Input IDs shape: {input_ids.shape}")
    
    # 生成回答
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            temperature=0,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"Full output: {outputs}")
    
    # 提取回答
    if prompt in outputs:
        answer = outputs.split(prompt)[-1].strip()
    else:
        answer = outputs.strip()
    
    print(f"Extracted answer: '{answer}'")
    
    if answer and answer != "":
        print("✅ 文本推理成功!")
    else:
        print("❌ 文本推理失败 - 输出为空")
    
    # 6. 测试图像推理
    print("6. 测试图像推理...")
    
    # 找一个存在的图像
    image_folder = "/perception-hl/zhuofan.xia/data/gqa/images"
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    if not image_files:
        print("❌ 没有找到图像文件")
        return
    
    test_image = os.path.join(image_folder, image_files[0])
    print(f"使用图像: {test_image}")
    
    # 加载图像
    image = Image.open(test_image).convert('RGB')
    
    # 构建图像对话
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\nWhat do you see in this image?")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(f"Image prompt: {prompt}")
    
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
            max_new_tokens=20,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"Image full output: {outputs}")
    
    # 提取回答
    if prompt in outputs:
        answer = outputs.split(prompt)[-1].strip()
    else:
        answer = outputs.strip()
    
    print(f"Image extracted answer: '{answer}'")
    
    if answer and answer != "":
        print("✅ 图像推理成功!")
    else:
        print("❌ 图像推理失败 - 输出为空")
    
    print("="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    test_simple_inference()
