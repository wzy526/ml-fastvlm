#!/usr/bin/env python3
"""
GQA调试测试脚本
用于验证模型推理是否正常工作
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
    model_name = get_model_name_from_path(model_path)
    if 'qwen' in model_name.lower():
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # 设置图像处理器
    try:
        image_processor = transformers.CLIPImageProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
    except OSError as e:
        if "preprocessor_config.json" in str(e):
            print("使用训练时的视觉编码器...")
            vision_tower_path = "/home/zhuofan.xia/gsva_pretrains/clip-vit-large-patch14-336"
            image_processor = transformers.CLIPImageProcessor.from_pretrained(
                vision_tower_path, trust_remote_code=True
            )
        else:
            raise e
    
    return model, tokenizer, image_processor


def test_single_inference(model, tokenizer, image_processor, image_path, question):
    """测试单个推理"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 构建对话
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
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
            answer = outputs.split(prompt)[-1].strip()
        else:
            answer = outputs.strip()
        
        print(f"Extracted answer: '{answer}'")
        return answer
        
    except Exception as e:
        print(f"推理错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # 测试配置
    model_path = "/perception-hl/zhuofan.xia/vlm_exps/textdat/tdat-7b-l0d32-s12g8z3"
    image_folder = "/perception-hl/zhuofan.xia/data/gqa/images"
    
    # 测试图像和问题
    test_image_id = "1216"  # 从您的结果文件中看到的图像ID
    test_question = "What color is the object?"
    
    print("="*80)
    print("GQA调试测试")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"图像ID: {test_image_id}")
    print(f"问题: {test_question}")
    print("="*80)
    
    # 加载模型
    print("加载模型...")
    model, tokenizer, image_processor = load_model_and_tokenizer(model_path)
    print("模型加载完成")
    
    # 测试推理
    image_path = os.path.join(image_folder, f"{test_image_id}.jpg")
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return
    
    print(f"图像路径: {image_path}")
    print("开始推理测试...")
    
    answer = test_single_inference(model, tokenizer, image_processor, image_path, test_question)
    
    if answer:
        print(f"✅ 推理成功! 答案: '{answer}'")
    else:
        print("❌ 推理失败!")


if __name__ == "__main__":
    main()
