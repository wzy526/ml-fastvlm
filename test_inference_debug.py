#!/usr/bin/env python3
"""
测试推理过程 - 诊断为什么准确率是0
"""

import os
import json
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as LlavaLlamaForCausalLM_LLaVA
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images

def load_model_and_tokenizer(model_path, vision_tower_path):
    """加载模型和tokenizer"""
    print("加载模型和tokenizer...")
    
    # 禁用torch初始化
    disable_torch_init()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 加载模型配置
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 修复decoder_config问题
    if hasattr(config, 'decoder_config') and isinstance(config.decoder_config, dict):
        from transformers import PretrainedConfig
        config.decoder_config = PretrainedConfig(**config.decoder_config)
    
    # 加载模型
    model = LlavaLlamaForCausalLM_LLaVA.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
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
        image_processor = CLIPImageProcessor.from_pretrained(model_path)
        print("使用模型路径的图像处理器")
    except OSError:
        print("模型路径没有图像处理器，使用vision_tower路径")
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower_path)
    
    return model, tokenizer, image_processor

def test_single_inference():
    """测试单个样本的推理过程"""
    print("开始测试单个样本推理...")
    
    # 模型路径
    model_path = "/ephstorage/vlm_exps/textdat/tdat-7b-l0d32-s12g8z3"
    vision_tower_path = "/home/zhuofan.xia/gsva_pretrains/clip-vit-large-patch14-336"
    
    # 加载模型
    model, tokenizer, image_processor = load_model_and_tokenizer(model_path, vision_tower_path)
    
    # 加载GQA数据
    with open('/perception-hl/zhuofan.xia/data/gqa/val_balanced_questions.json', 'r') as f:
        data = json.load(f)
    
    # 获取第一个样本
    first_qid = list(data.keys())[0]
    sample = data[first_qid]
    
    print(f"测试样本:")
    print(f"  Question ID: {first_qid}")
    print(f"  Image ID: {sample['imageId']}")
    print(f"  Question: {sample['question']}")
    print(f"  Answer: {sample['answer']}")
    
    # 查找图像文件
    image_folder = '/perception-hl/zhuofan.xia/data/gqa/images'
    image_id = sample['imageId']
    
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
        print(f"错误: 找不到图像文件 {image_id}")
        return
    
    print(f"找到图像: {image_path}")
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    print(f"图像尺寸: {image.size}")
    
    # 构建对话
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{sample['question']}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(f"完整prompt:")
    print(prompt)
    print("="*50)
    
    # 处理输入
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).cuda()
    
    print(f"输入token数量: {input_ids.shape[1]}")
    
    # 处理图像
    image_tensor = process_images([image], image_processor, model.config)[0]
    print(f"图像tensor形状: {image_tensor.shape}")
    
    # 生成回答
    print("开始生成回答...")
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
    print(f"完整输出:")
    print(outputs)
    print("="*50)
    
    # 提取回答
    if prompt in outputs:
        answer = outputs.split(prompt)[-1].strip()
    else:
        answer = outputs.strip()
    
    if answer:
        lines = answer.split('\n')
        if lines:
            answer = lines[0].strip()
    
    print(f"提取的回答: '{answer}'")
    print(f"正确答案: '{sample['answer']}'")
    print(f"是否匹配: {answer.lower().strip() == sample['answer'].lower().strip()}")

if __name__ == "__main__":
    test_single_inference()
