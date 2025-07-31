#!/usr/bin/env python3
#
# LLaVA-FastViTHD-0.5B Stage 2 TTFT 测试脚本
# 基于LLaVA架构的FastVLM-0.5B Stage 2模型：
# - Architecture: LLaVA (Large Language and Vision Assistant)
# - Vision Encoder: FastViTHD
# - LLM: LLaVA (Llama-based)
# - Input Res: 1024
# - #Visual Tokens: 256
# - Vis. Enc. Size: 125M
# - Model Size: 0.5B parameters
# - Stage: 2 
#
import os
import time
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM
import transformers


class TestDataset(Dataset):
    """TTFT test dataset for LLaVA-FastViTHD-0.5B Stage 2"""
    
    def __init__(self, data_path, image_folder, tokenizer, image_processor, max_samples=None):
        """
        Args:
            data_path: GQA数据文件路径 (.json)
            image_folder: 图像文件夹路径
            tokenizer: 分词器
            image_processor: 图像处理器
            max_samples: 最大样本数
        """
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        # load GQA data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # convert to list format
        self.samples = []
        for qid, item in self.data.items():
            if max_samples and len(self.samples) >= max_samples:
                break
            self.samples.append({
                'qid': qid,
                'question': item['question'],
                'imageId': item['imageId'],
                'answer': item.get('answer', ''),
                'fullAnswer': item.get('fullAnswer', '')
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = os.path.join(self.image_folder, f"{sample['imageId']}.jpg")
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.resize_for_fastvlm(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (1024, 1024), color='gray')
        
        return {
            'qid': sample['qid'],
            'question': sample['question'],
            'image': image,
            'answer': sample['answer'],
            'fullAnswer': sample['fullAnswer']
        }
    
    def resize_for_fastvlm(self, image):
        """resize image to FastVLM target resolution (1024x1024)"""
        target_size = 1024
        
        w, h = image.size
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        
        # resize image
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # if image is not square, pad it
        if new_w != target_size or new_h != target_size:
            # create white background
            background = Image.new('RGB', (target_size, target_size), (255, 255, 255))
            # paste resized image to center
            x = (target_size - new_w) // 2
            y = (target_size - new_h) // 2
            background.paste(image, (x, y))
            image = background
        
        return image


def setup_distributed():
    """setup distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return None, None, None
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    
    return rank, world_size, local_rank


def load_model(model_path, device):
    """load LLaVA-FastViTHD-0.5B Stage 2 model"""
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    
    print(f"Loading LLaVA-FastViTHD-0.5B Stage 2 model: {model_name}")
    print(f"Architecture: LLaVA with FastViTHD + Llama")
    print(f"Expected config: FastViTHD + Llama, 1024x1024 input, 256 visual tokens")
    print(f"Model size: 0.5B parameters, Stage 2")

    compute_dtype = torch.float16 # according to llava training setting
    
    config = transformers.AutoConfig.from_pretrained(model_path)
    if config.model_type == "llava_qwen2":
        print(f"Loading LlavaQwen2ForCausalLM model (Qwen2 backbone)")
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            device_map={"": device}
        )
    else:
        print(f"Loading LlavaLlamaForCausalLM model (Llama backbone)")
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            device_map={"": device}
        )
    
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        use_fast=False,
    )
    
    # pad token
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.legacy = False
    
    # load FastViTHD vision tower
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        image_processor = vision_tower.image_processor
        print(f"FastViTHD vision tower loaded: {type(vision_tower).__name__}")
        
        # verify vision encoder config
        if hasattr(vision_tower, 'config'):
            print(f"Vision encoder config: {vision_tower.config}")
    else:
        image_processor = None
        print("Warning: No vision tower found")
    
    # inference mode
    model.eval()
    model.config.use_cache = True
    
    print(f"LLaVA model loaded successfully")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model device: {next(model.parameters()).device}")
    
    return tokenizer, model, image_processor


def measure_fastvlm_ttft(model, tokenizer, image_processor, sample, conv_mode="llava_v1"):
    """measure TTFT for LLaVA-FastViTHD-0.5B Stage 2"""
    
    # 数据预处理不计入TTFT时间
    # prompt
    qs = sample['question']
    conv = conv_templates[conv_mode].copy()
    
    # add image token
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # pad token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    input_ids = input_ids.to(model.device)
    
    # process image to target resolution
    if image_processor is not None:
        image = sample['image']
        if image.size != (1024, 1024):
            image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).half().to(model.device)
        image_size = image.size
    else:
        image_tensor = None
        image_size = None
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # TTFT测量开始 - 只测量模型推理时间
    start_time = time.time()
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size] if image_size else None,
            max_new_tokens=1,  # only generate first token
            use_cache=True,
            do_sample=False,  # greedy decoding for consistent TTFT
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id
        )
    
    torch.cuda.synchronize()
    end_time = time.time()
    ttft = (end_time - start_time) * 1000 # mili sec
    
    return ttft


def test_fastvlm_ttft(args):
    """test TTFT performance for LLaVA-FastViTHD-0.5B Stage 2"""
    
    # setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if local_rank is not None else "cuda:0"
    
    if rank is None:
        rank = 0
    if world_size is None:
        world_size = 1
    if local_rank is None:
        local_rank = 0
    
    if rank == 0:
        print("="*60)
        print("LLaVA-FastViTHD-0.5B Stage 2 TTFT test")
        print("Config: FastViTHD + Llama, 1024x1024, 256 visual tokens")
        print("Model: 0.5B parameters, Stage 2")
        print("="*60)
    
    tokenizer, model, image_processor = load_model(args.model_path, device)

    dataset = TestDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_samples=args.max_samples
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None
    

    def custom_collate_fn(batch):
        # 由于batch_size=1，直接返回第一个元素
        return batch[0]
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  #一个图片一个图片处理
        sampler=sampler,
        num_workers=0,  # 设置为0避免多进程问题
        pin_memory=False, 
        collate_fn=custom_collate_fn
    )
    
    # acculate latency
    accumulated_latency = 0.0
    running_samples = 0
    
    if rank == 0:
        print(f"Start LLaVA-FastViTHD-0.5B Stage 2 TTFT test, total {len(dataset)} samples")
        print("Using dist.all_reduce(sum) to calculate average TTFT")
    
    for i, sample in enumerate(tqdm(dataloader, disable=rank != 0)):
        # sample已经是单个样本，不需要进一步处理
        
        try:
            ttft = measure_fastvlm_ttft(model, tokenizer, image_processor, sample, args.conv_mode)
            accumulated_latency += ttft
            running_samples += 1
            
            if rank == 0 and i % 50 == 0:
                current_avg = accumulated_latency / running_samples
                print(f"Sample {i}: current average TTFT = {current_avg:.2f}ms")
                
        except Exception as e:
            if rank == 0:
                print(f"Error processing sample {i}: {e}")
            continue
    
    # use all_reduce to calculate global accumulated latency and samples
    accumulated_latency_tensor = torch.tensor(accumulated_latency, device=device)
    running_samples_tensor = torch.tensor(running_samples, device=device)
    
    if world_size > 1:
        dist.all_reduce(accumulated_latency_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_samples_tensor, op=dist.ReduceOp.SUM)
    
    # calculate global average TTFT
    global_accumulated_latency = accumulated_latency_tensor.item()
    global_running_samples = running_samples_tensor.item()
    avg_ttft = global_accumulated_latency / global_running_samples if global_running_samples > 0 else 0.0
    
    if rank == 0:
        print("\n" + "="*60)
        print("LLaVA-FastViTHD-0.5B Stage 2 TTFT test results")
        print("="*60)
        print(f"Global samples: {global_running_samples}")
        print(f"Global accumulated latency: {global_accumulated_latency:.2f}ms")
        print(f"Global average TTFT: {avg_ttft:.2f}ms")
        
        print(f"\nCalculation method:")
        print(f"- Per card accumulated latency: {accumulated_latency:.2f}ms")
        print(f"- Per card samples: {running_samples}")
        print(f"- Using dist.all_reduce(sum) to sum")
        print(f"- Global average = total accumulated latency / total samples")

        if args.output_file:
            results = {
                'model_config': 'LLaVA_FastViTHD_0.5B_Stage2',
                'vision_encoder': 'FastViTHD',
                'llm': 'Llama',
                'input_resolution': '1024x1024',
                'visual_tokens': 256,
                'model_size': '0.5B',
                'stage': 2,
                'model_path': args.model_path,
                'data_path': args.data_path,
                'total_samples': global_running_samples,
                'accumulated_latency_ms': global_accumulated_latency,
                'avg_ttft_ms': avg_ttft,
                'calculation_method': 'dist.all_reduce(sum)',
                'world_size': world_size,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA-FastViTHD-0.5B Stage 2 TTFT test")
    parser.add_argument("--model-path", type=str, required=True, help="LLaVA model path")
    parser.add_argument("--data-path", type=str, required=True, help="GQA data file path")
    parser.add_argument("--image-folder", type=str, required=True, help="GQA image folder path")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation mode")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--output-file", type=str, default="llava_fastvithd_0. 5b_stage2_ttft_results.json", help="Output results file")
    
    args = parser.parse_args()
    
    test_fastvlm_ttft(args) 