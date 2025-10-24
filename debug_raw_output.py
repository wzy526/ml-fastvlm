#!/usr/bin/env python3
"""
调试DAT模型的原始输出
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import transformers

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math

def debug_single_sample():
    # 加载模型
    disable_torch_init()
    model_path = "/data/checkpoints/weilai/tdat-7b-l0d32-s12g8z3"
    model_name = get_model_name_from_path(model_path)
    
    # 检查是否是DAT模型
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    is_dat_model = hasattr(config, 'architectures') and 'LlavaLlamaDATForCausalLM' in config.architectures
    
    if is_dat_model:
        print("检测到DAT模型，使用GQA兼容的加载方式...")
        from llava.eval.eval_gqa import load_model_and_tokenizer
        model, tokenizer, image_processor = load_model_and_tokenizer(model_path, device='cuda')
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    # 加载问题数据
    questions = [json.loads(q) for q in open("/data/textvqa/llava_textvqa_val_v051_ocr.jsonl", "r")]
    
    # 找到问题样本
    target_question = "What is the advertisement in the white board?"
    target_sample = None
    for q in questions:
        if target_question in q["text"]:
            target_sample = q
            break
    
    if not target_sample:
        print("未找到目标样本")
        return
    
    print(f"找到目标样本: {target_sample['question_id']}")
    print(f"问题: {target_sample['text']}")
    
    # 处理图像
    image_file = target_sample["image"]
    qs = target_sample["text"]
    cur_prompt = qs
    
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print(f"\n完整prompt:")
    print(prompt)
    print(f"\n{'='*50}")

    # 处理图像
    image = Image.open(os.path.join("/data/textvqa/train_images", image_file)).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).to(device='cuda', non_blocking=True)

    # 生成输出
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=128,
            use_cache=True)

    # 解码输出
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    print(f"\n原始完整输出:")
    print(repr(outputs))
    print(f"\n{'='*50}")
    
    # 提取回答部分
    if cur_prompt in outputs:
        extracted = outputs.split(cur_prompt)[-1].strip()
        print(f"\n提取的回答部分:")
        print(repr(extracted))
    else:
        print(f"\n未找到问题在输出中，直接使用输出:")
        print(repr(outputs.strip()))

if __name__ == "__main__":
    debug_single_sample()
