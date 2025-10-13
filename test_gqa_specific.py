#!/usr/bin/env python3
"""
测试GQA特定的推理逻辑
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


def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    disable_torch_init()
    
    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    # 修复配置
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
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 初始化视觉编码器
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
            vision_tower_path = "/home/zhuofan.xia/gsva_pretrains/clip-vit-large-patch14-336"
            image_processor = transformers.CLIPImageProcessor.from_pretrained(
                vision_tower_path, trust_remote_code=True
            )
        else:
            raise e
    
    return model, tokenizer, image_processor


def test_gqa_sample(model, tokenizer, image_processor, sample, image_folder):
    """测试单个GQA样本"""
    try:
        # 加载图像
        image_path = os.path.join(image_folder, f"{sample['imageId']}.jpg")
        if not os.path.exists(image_path):
            return None, "Image not found"
        
        image = Image.open(image_path).convert('RGB')
        
        # 构建对话 - 使用llava_v1格式
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{sample['question']}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        print(f"Question: {sample['question']}")
        print(f"Expected answer: {sample['answer']}")
        print(f"Prompt: {prompt}")
        
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
        print(f"Full output: {outputs}")
        
        # 提取回答
        if prompt in outputs:
            outputs = outputs.split(prompt)[-1].strip()
        else:
            outputs = outputs.strip()
        
        # 清理输出
        if outputs:
            lines = outputs.split('\n')
            if lines:
                outputs = lines[0].strip()
        
        print(f"Extracted answer: '{outputs}'")
        return outputs, None
        
    except Exception as e:
        print(f"推理错误: {e}")
        return None, str(e)


def main():
    # 加载模型
    model_path = "/perception-hl/zhuofan.xia/vlm_exps/textdat/tdat-7b-l0d32-s12g8z3"
    image_folder = "/perception-hl/zhuofan.xia/data/gqa/images"
    data_path = "/perception-hl/zhuofan.xia/data/gqa/val_balanced_questions.json"
    
    print("="*80)
    print("GQA特定测试")
    print("="*80)
    
    # 加载模型
    print("加载模型...")
    model, tokenizer, image_processor = load_model_and_tokenizer(model_path)
    print("模型加载完成")
    
    # 加载GQA数据
    print("加载GQA数据...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 转换为列表格式
    samples = []
    for qid, item in data.items():
        samples.append({
            'questionId': qid,
            'question': item['question'],
            'imageId': item['imageId'],
            'answer': item['answer']
        })
    
    print(f"加载了 {len(samples)} 个样本")
    
    # 测试前几个样本
    print("测试前3个样本...")
    for i in range(min(3, len(samples))):
        print(f"\n--- 样本 {i+1} ---")
        sample = samples[i]
        
        pred, error = test_gqa_sample(model, tokenizer, image_processor, sample, image_folder)
        
        if error:
            print(f"❌ 错误: {error}")
        else:
            print(f"✅ 预测: '{pred}'")
            print(f"✅ 真实: '{sample['answer']}'")
            print(f"✅ 匹配: {pred.lower().strip() == sample['answer'].lower().strip()}")
    
    print("="*80)
    print("测试完成")
    print("="*80)


if __name__ == "__main__":
    main()
