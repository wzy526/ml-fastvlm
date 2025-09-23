#!/usr/bin/env python3
#
# FastVLM TTFT 测试脚本 - 支持多种视觉编码器
# 基于LLaVA架构的多视觉编码器模型：
# - Architecture: LLaVA (Large Language and Vision Assistant)
# - Vision Encoder: 支持 FastViTHD, CLIP-ViT-L/14-336px, CLIP-ViT-L/14-336px-S2, CLIP-ViT-L/14-336px-LLaVA16 等
# - LLM: LLaVA (Llama-based)
# - Input Res: 可配置 (336, 512, 768, 1024, 1536, 2048)
# - #Visual Tokens: 可配置 (根据编码器和分辨率自动计算或手动指定)
# - Model Size: 可配置
# - Stage: 可配置 
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


def calculate_visual_tokens(resolution, vision_encoder, patch_size=None):
    """
    根据分辨率和视觉编码器类型计算visual token数量
    
    Args:
        resolution (int): 输入分辨率
        vision_encoder (str): 视觉编码器类型
        patch_size (int, optional): patch大小，如果为None则使用默认值
    
    Returns:
        int: visual token数量
    """
    if vision_encoder == "fastvithd":
        return 256
    elif vision_encoder == "clip":
        if patch_size is None:
            patch_size = 14
        num_patches = (resolution // patch_size) ** 2
        return num_patches
    elif vision_encoder == "clip_s2":
        if patch_size is None:
            patch_size = 14
        num_patches = (resolution // patch_size) ** 2
        return num_patches
    elif vision_encoder == "clip_llava16":
        # LLaVA-1.6分块方式：每个patch经过CLIP产生visual tokens
        # 每个336x336的patch产生 (336/14)^2 = 576个tokens
        if resolution == 1008:
            # 1个全局patch + 9个局部patches = 10个patches
            # 10 × 576 = 5760 tokens
            return 10 * 576
        elif resolution == 672:
            # 1个全局patch + 4个局部patches = 5个patches  
            # 5 × 576 = 2880 tokens
            return 5 * 576
        elif resolution == 336:
            # 只有1个全局patch
            # 1 × 576 = 576 tokens
            return 1 * 576
        else:
            if patch_size is None:
                patch_size = 336
            grid_size = resolution // patch_size
            num_patches = 1 + (grid_size ** 2)
            return num_patches * 576
    else:
        if patch_size is None:
            patch_size = 14
        num_patches = (resolution // patch_size) ** 2
        return num_patches


class TestDataset(Dataset):
    """TTFT test dataset for LLaVA models with different vision encoders"""
    
    def __init__(self, data_path, image_folder, tokenizer, image_processor, max_samples=None, resolution=1024):
        """
        Args:
            data_path: GQA数据文件路径 (.json)
            image_folder: 图像文件夹路径
            tokenizer: 分词器
            image_processor: 图像处理器
            max_samples: 最大样本数
            resolution: 目标分辨率 (默认1024)
        """
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.resolution = resolution
        
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
            image = Image.new('RGB', (self.resolution, self.resolution), color='gray')
        
        return {
            'qid': sample['qid'],
            'question': sample['question'],
            'image': image,
            'answer': sample['answer'],
            'fullAnswer': sample['fullAnswer']
        }
    
    def resize_for_fastvlm(self, image):
        """resize image to target resolution for vision encoder"""
        target_size = self.resolution
        
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


def load_model(model_path, device, resolution=1024, vision_encoder="fastvithd"):
    """load LLaVA model with specified vision encoder"""
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    
    # 先加载模型以检测backbone类型，稍后更新打印信息
    pass
    
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Vision Encoder: {vision_encoder}")

    compute_dtype = torch.float16 # according to llava training setting
    
    config = transformers.AutoConfig.from_pretrained(model_path)
    
    # check llm backbone type
    if config.model_type == "llava_qwen2":
        llm_backbone = "Qwen2"
        print(f"Loading LlavaQwen2ForCausalLM model (Qwen2 backbone)")
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            device_map="auto"  
        )
    else:
        llm_backbone = "Llama"
        print(f"Loading LlavaLlamaForCausalLM model (Llama backbone)")
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=compute_dtype,
            device_map="auto" 
        )
    
    # 根据vision_encoder类型更新打印信息
    if vision_encoder == "fastvithd":
        print(f"Loading LLaVA-FastViTHD model: {model_name}")
        print(f"Architecture: LLaVA with FastViTHD + {llm_backbone}")
        print(f"Expected config: FastViTHD + {llm_backbone}, {resolution}x{resolution} input, 256 visual tokens")
    elif vision_encoder == "clip":
        print(f"Loading LLaVA-CLIP model: {model_name}")
        print(f"Architecture: LLaVA with CLIP-ViT-L/14-336px + {llm_backbone}")
        print(f"Expected config: CLIP-ViT-L/14-336px + {llm_backbone}, {resolution}x{resolution} input")
    elif vision_encoder == "clip_s2":
        print(f"Loading LLaVA-CLIP-S2 model: {model_name}")
        print(f"Architecture: LLaVA with CLIP-ViT-L/14-336px-S2 + {llm_backbone}")
        print(f"Expected config: CLIP-ViT-L/14-336px-S2 + {llm_backbone}, {resolution}x{resolution} input (multi-scale)")
    elif vision_encoder == "clip_llava16":
        print(f"Loading LLaVA-CLIP-LLaVA16 model: {model_name}")
        print(f"Architecture: LLaVA with CLIP-ViT-L/14-336px-LLaVA16 + {llm_backbone}")
        print(f"Expected config: CLIP-ViT-L/14-336px-LLaVA16 + {llm_backbone}, {resolution}x{resolution} input (LLaVA-1.6 patching)")
        
        # llava16 anyres config修改
        model.config.image_aspect_ratio = 'fixed_hr'
        model.config.image_grid_pinpoints = None    
        print(f"Updated config - image_aspect_ratio: {model.config.image_aspect_ratio}")
        print(f"Updated config - image_grid_pinpoints: {model.config.image_grid_pinpoints}")
    else:
        print(f"Loading LLaVA model with {vision_encoder}: {model_name}")
        print(f"Architecture: LLaVA with {vision_encoder} + {llm_backbone}")
        print(f"Expected config: {vision_encoder} + {llm_backbone}, {resolution}x{resolution} input")
    
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
    
        # load vision tower
    if hasattr(model, 'get_vision_tower'):
        vision_tower = model.get_vision_tower()
        print(f"Vision tower loaded: {type(vision_tower).__name__}")
        
        if not vision_tower.is_loaded:
            print("Loading vision tower...")
            vision_tower.load_model()
        
        if hasattr(vision_tower, 'vision_tower') and hasattr(vision_tower.vision_tower, 'device') and vision_tower.vision_tower.device != device:
            print(f"Moving vision tower from {vision_tower.vision_tower.device} to {device}")
            vision_tower.vision_tower = vision_tower.vision_tower.to(device)
        
        # 获取image_processor
        if hasattr(vision_tower, 'image_processor'):
            image_processor = vision_tower.image_processor
            # 添加调试信息显示实际使用的分辨率
            if vision_encoder == "fastvithd":
                print(f"FastViTHD actual image size: {image_processor.size}")
                print(f"FastViTHD actual crop size: {image_processor.crop_size}")
                if hasattr(vision_tower, 'config'):
                    print(f"FastViTHD config image_size: {vision_tower.config.get('image_cfg', {}).get('image_size', 'N/A')}")
        else:
            print("Warning: Vision tower has no image_processor attribute")
            image_processor = None
        
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


def measure_fastvlm_ttft(model, tokenizer, image_processor, sample, conv_mode="llava_v1", resolution=1024, vision_encoder="fastvithd"):
    """measure TTFT for LLaVA model with specified vision encoder"""
    
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
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor[None, ...].half().to(model.device)
        image_size = image.size
        
        if image_tensor.device != model.device:
            image_tensor = image_tensor.to(model.device)
    else:
        image_tensor = None
        image_size = None
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # TTFT测量开始 - 只测量模型推理时间
    start_time = time.time()
    
    with torch.inference_mode():
        if image_tensor is not None and image_tensor.device != model.device:
            image_tensor = image_tensor.to(model.device)
        model.generate(
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
    """test TTFT performance for LLaVA model with specified vision encoder"""
    
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
        if args.vision_encoder == "fastvithd":
            print(f"LLaVA-FastViTHD TTFT test - {args.resolution}x{args.resolution}")
            print(f"Config: FastViTHD + [LLM Backbone], {args.resolution}x{args.resolution}, 256 visual tokens")
        elif args.vision_encoder == "clip":
            print(f"LLaVA-CLIP TTFT test - {args.resolution}x{args.resolution}")
            print(f"Config: CLIP-ViT-L/14-336px + [LLM Backbone], {args.resolution}x{args.resolution}")
        elif args.vision_encoder == "clip_s2":
            print(f"LLaVA-CLIP-S2 TTFT test - {args.resolution}x{args.resolution}")
            print(f"Config: CLIP-ViT-L/14-336px-S2 + [LLM Backbone], {args.resolution}x{args.resolution} (multi-scale)")
        elif args.vision_encoder == "clip_llava16":
            print(f"LLaVA-CLIP-LLaVA16 TTFT test - {args.resolution}x{args.resolution}")
            print(f"Config: CLIP-ViT-L/14-336px-LLaVA16 + [LLM Backbone], {args.resolution}x{args.resolution} (LLaVA-1.6 patching)")
        else:
            print(f"LLaVA-{args.vision_encoder.upper()} TTFT test - {args.resolution}x{args.resolution}")
            print(f"Config: {args.vision_encoder.upper()} + [LLM Backbone], {args.resolution}x{args.resolution}")
        print("="*60)
    
    tokenizer, model, image_processor = load_model(args.model_path, device, args.resolution, args.vision_encoder)

    dataset = TestDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_samples=args.max_samples,
        resolution=args.resolution
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
        num_workers=10, 
        pin_memory=False, 
        collate_fn=custom_collate_fn
    )
    
    # accumulated latency
    accumulated_latency = 0.0
    running_samples = 0
    
    # warm up samples
    warmup_samples = 100
    
    if rank == 0:
        if args.vision_encoder == "fastvithd":
            print(f"Start LLaVA-FastViTHD TTFT test, total {len(dataset)} samples")
        elif args.vision_encoder == "clip":
            print(f"Start LLaVA-CLIP TTFT test, total {len(dataset)} samples")
        else:
            print(f"Start LLaVA-{args.vision_encoder.upper()} TTFT test, total {len(dataset)} samples")
        print("Using dist.all_reduce(sum) to calculate average TTFT")
        print(f"Warmup phase: first {warmup_samples} samples will not be counted in TTFT measurement")
    
    if rank == 0:
        print(f"Starting warmup with {warmup_samples} samples...")
    
    warmup_count = 0
    for i, sample in enumerate(tqdm(dataloader, disable=rank != 0)):
        try:
            if i < warmup_samples:
                # warm up phase - not count time
                _ = measure_fastvlm_ttft(model, tokenizer, image_processor, sample, args.conv_mode, args.resolution, args.vision_encoder)
                warmup_count += 1
                if rank == 0 and i % 20 == 0:
                    print(f"Warmup sample {i+1}/{warmup_samples}")
            else:
                # test phase - count time
                ttft = measure_fastvlm_ttft(model, tokenizer, image_processor, sample, args.conv_mode, args.resolution, args.vision_encoder)
                accumulated_latency += ttft
                running_samples += 1
                
                if rank == 0 and (i - warmup_samples) % 50 == 0:
                    current_avg = accumulated_latency / running_samples
                    print(f"Test sample {i - warmup_samples}: current average TTFT = {current_avg:.2f}ms")
                
        except Exception as e:
            if rank == 0:
                if i < warmup_samples:
                    print(f"Error processing warmup sample {i}: {e}")
                else:
                    print(f"Error processing test sample {i - warmup_samples}: {e}")
            continue
    
    if rank == 0:
        print(f"Warmup completed: {warmup_count} samples")
        print(f"Actual TTFT measurement: {running_samples} samples")
    
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
        if args.vision_encoder == "fastvithd":
            print(f"LLaVA-FastViTHD TTFT test results - {args.resolution}x{args.resolution}")
        elif args.vision_encoder == "clip":
            print(f"LLaVA-CLIP TTFT test results - {args.resolution}x{args.resolution}")
        elif args.vision_encoder == "clip_s2":
            print(f"LLaVA-CLIP-S2 TTFT test results - {args.resolution}x{args.resolution}")
        elif args.vision_encoder == "clip_llava16":
            print(f"LLaVA-CLIP-LLaVA16 TTFT test results - {args.resolution}x{args.resolution}")
        else:
            print(f"LLaVA-{args.vision_encoder.upper()} TTFT test results - {args.resolution}x{args.resolution}")
        print("="*60)
        print(f"Warmup samples: {warmup_samples}")
        print(f"Global samples: {global_running_samples}")
        print(f"Global accumulated latency: {global_accumulated_latency:.2f}ms")
        print(f"Global average TTFT: {avg_ttft:.2f}ms")
        
        print(f"\nCalculation method:")
        print(f"- Warmup samples: {warmup_samples} (not counted)")
        print(f"- Per card accumulated latency: {accumulated_latency:.2f}ms")
        print(f"- Per card samples: {running_samples}")
        print(f"- Using dist.all_reduce(sum) to sum")
        print(f"- Global average = total accumulated latency / total samples")

        if args.output_file:
            if args.visual_tokens is not None:
                # 如果用户指定了visual token数量，直接使用
                visual_tokens = args.visual_tokens
            else:
                # 否则根据分辨率和vision encoder自动计算
                visual_tokens = calculate_visual_tokens(args.resolution, args.vision_encoder, args.patch_size)
            
            # 设置模型配置名称
            if args.vision_encoder == "fastvithd":
                model_config = 'LLaVA_FastViTHD_0.5B_Stage2'
            elif args.vision_encoder == "clip":
                model_config = 'LLaVA_CLIP_ViT_L_14_336px'
            elif args.vision_encoder == "clip_s2":
                model_config = 'LLaVA_CLIP_ViT_L_14_336px_S2'
            elif args.vision_encoder == "clip_llava16":
                model_config = 'LLaVA_CLIP_ViT_L_14_336px_LLaVA16'
            else:
                model_config = f'LLaVA_{args.vision_encoder.upper()}'
            
            # 用来保存
            config = transformers.AutoConfig.from_pretrained(args.model_path)
            llm_backbone = "Qwen2" if config.model_type == "llava_qwen2" else "Llama"
            
            results = {
                'model_config': model_config,
                'vision_encoder': args.vision_encoder,
                'llm': llm_backbone,
                'input_resolution': f'{args.resolution}x{args.resolution}',
                'visual_tokens': visual_tokens,
                'model_size': '0.5B',
                'stage': 2,
                'model_path': args.model_path,
                'data_path': args.data_path,
                'warmup_samples': warmup_samples,
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
    parser.add_argument("--resolution", type=int, default=1024, choices=[224, 256, 336, 384, 448, 512, 672, 768, 1008, 1024, 1152, 1344, 1536], 
                       help="Input resolution (224, 256, 336, 384, 448, 512, 672, 768, 1008, 1024, 1152, 1344, or 1536)")
    parser.add_argument("--output-file", type=str, default=None, help="Output results file")
    parser.add_argument("--vision-encoder", type=str, default="fastvithd", choices=["fastvithd", "clip", "clip_s2", "clip_llava16"],
                       help="Vision encoder to use (fastvithd, clip, clip_s2, or clip_llava16)")
    parser.add_argument("--visual-tokens", type=int, default=None, 
                       help="Number of visual tokens (if None, will be calculated automatically based on resolution and vision encoder)")
    parser.add_argument("--patch-size", type=int, default=None,
                       help="Patch size for visual token calculation (if None, will use default values for each encoder)")
    
    args = parser.parse_args()
    
    if args.output_file is None:
        if args.visual_tokens is not None:
            visual_tokens_str = f"vt{args.visual_tokens}"
        else:
            visual_tokens = calculate_visual_tokens(args.resolution, args.vision_encoder, args.patch_size)
            visual_tokens_str = f"vt{visual_tokens}"
        
        args.output_file = f"ttft_test_results_{args.vision_encoder}_{args.resolution}x{args.resolution}_{visual_tokens_str}.json"
    
    test_fastvlm_ttft(args) 