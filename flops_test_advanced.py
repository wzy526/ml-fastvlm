#!/usr/bin/env python3
#
# FastVLM FLOPs 计算脚本 (高级版本) - 支持多种视觉编码器
# 使用PyTorch FLOPs计算库获得更精确的结果
# 基于LLaVA架构的多视觉编码器模型：
# - Architecture: LLaVA (Large Language and Vision Assistant)
# - Vision Encoder: 支持 FastViTHD, CLIP-ViT-L/14-336px, CLIP-ViT-L/14-336px-S2, CLIP-ViT-L/14-336px-LLaVA16 等
# - LLM: LLaVA (Llama-based/Qwen2-based)
# - Input Res: 可配置 (336, 512, 768, 1024, 1536, 2048)
# - #Visual Tokens: 可配置 (根据编码器和分辨率自动计算或手动指定)
# - Model Size: 可配置
# - Stage: 可配置 
#
import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import json
import time
from typing import Dict, Any, Optional

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    try:
        from thop import profile
        THOP_AVAILABLE = True
        FVCORE_AVAILABLE = False
    except ImportError:
        FVCORE_AVAILABLE = False
        THOP_AVAILABLE = False
        print("警告: 未安装FLOPs计算库。请安装 fvcore 或 thop: pip install fvcore thop")

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


class AdvancedFLOPsCalculator:
    """高级VLM模型FLOPs计算器"""
    
    def __init__(self, model, tokenizer, image_processor, vision_encoder="fastvithd"):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.vision_encoder = vision_encoder
        
        # FLOPs结果
        self.encoder_flops = 0
        self.connector_flops = 0
        self.tokenizer_flops = 0
        self.llm_flops = 0
        
    def calculate_encoder_flops_with_fvcore(self, image_tensor):
        """使用fvcore计算视觉编码器FLOPs"""
        if not FVCORE_AVAILABLE:
            return self.calculate_encoder_flops_manual(image_tensor)
        
        try:
            # 获取视觉塔
            vision_tower = self.model.get_vision_tower()
            
            with FlopCountMode(self.model) as flop_counter:
                # 前向传播
                with torch.no_grad():
                    vision_outputs = vision_tower(image_tensor)
                
                # 获取FLOPs
                flops = flop_counter.get_total_flops()
                return flops
        except Exception as e:
            print(f"fvcore计算失败: {e}")
            return self.calculate_encoder_flops_manual(image_tensor)
    
    def calculate_encoder_flops_with_thop(self, image_tensor):
        """使用thop计算视觉编码器FLOPs"""
        if not THOP_AVAILABLE:
            return self.calculate_encoder_flops_manual(image_tensor)
        
        try:
            # 获取视觉塔
            vision_tower = self.model.get_vision_tower()
            
            # 使用thop计算FLOPs
            flops, params = profile(vision_tower, inputs=(image_tensor,), verbose=False)
            return flops
        except Exception as e:
            print(f"thop计算失败: {e}")
            return self.calculate_encoder_flops_manual(image_tensor)
    
    def calculate_encoder_flops_manual(self, image_tensor):
        """手动计算视觉编码器FLOPs"""
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        
        if self.vision_encoder == "fastvithd":
            # FastViTHD的FLOPs计算
            embed_dim = 3072
            num_layers = 12
            patch_size = 64
            num_patches = (H // patch_size) * (W // patch_size)
            
            # 自注意力的FLOPs: 4 * embed_dim * embed_dim * num_patches
            attention_flops = 4 * embed_dim * embed_dim * num_patches
            
            # FFN的FLOPs: 8 * embed_dim * embed_dim * num_patches
            ffn_flops = 8 * embed_dim * embed_dim * num_patches
            
            # 总FLOPs
            total_flops = num_layers * (attention_flops + ffn_flops)
            
            return total_flops
            
        elif self.vision_encoder == "clip":
            # CLIP的FLOPs计算
            embed_dim = 1024
            num_layers = 24
            patch_size = 14
            num_patches = (H // patch_size) * (W // patch_size)
            
            # 自注意力的FLOPs
            attention_flops = 4 * embed_dim * embed_dim * num_patches
            
            # FFN的FLOPs
            ffn_flops = 8 * embed_dim * embed_dim * num_patches
            
            # 总FLOPs
            total_flops = num_layers * (attention_flops + ffn_flops)
            
            return total_flops
            
        else:
            # 默认估算
            return H * W * 3 * 1000
    
    def calculate_connector_flops(self, visual_features):
        """计算连接器(投影层)FLOPs"""
        try:
            if hasattr(self.model, 'get_model') and hasattr(self.model.get_model(), 'mm_projector'):
                projector = self.model.get_model().mm_projector
                
                if FVCORE_AVAILABLE:
                    with FlopCountMode(self.model) as flop_counter:
                        with torch.no_grad():
                            _ = projector(visual_features)
                        flops = flop_counter.get_total_flops()
                        return flops
                elif THOP_AVAILABLE:
                    flops, _ = profile(projector, inputs=(visual_features,), verbose=False)
                    return flops
                else:
                    # 手动计算
                    if hasattr(projector, 'weight'):
                        in_features = projector.weight.shape[1]
                        out_features = projector.weight.shape[0]
                        flops = in_features * out_features * visual_features.shape[1]
                        return flops
        except Exception as e:
            print(f"连接器FLOPs计算失败: {e}")
        
        # 默认估算
        return visual_features.shape[0] * visual_features.shape[1] * 1000
    
    def calculate_llm_flops(self, input_ids, visual_tokens_count):
        """计算LLM的FLOPs"""
        try:
            # 创建完整的输入序列
            seq_len = input_ids.shape[1] + visual_tokens_count
            
            if FVCORE_AVAILABLE:
                with FlopCountMode(self.model) as flop_counter:
                    with torch.no_grad():
                        # 创建虚拟的visual tokens
                        visual_tokens = torch.randn(1, visual_tokens_count, self.model.config.hidden_size)
                        
                        # 合并输入
                        combined_input = torch.cat([visual_tokens, input_ids.float().unsqueeze(-1)], dim=1)
                        
                        # 前向传播
                        _ = self.model(inputs_embeds=combined_input)
                    
                    flops = flop_counter.get_total_flops()
                    return flops
            elif THOP_AVAILABLE:
                # 创建虚拟输入
                dummy_input = torch.randn(1, seq_len, self.model.config.hidden_size)
                flops, _ = profile(self.model, inputs=(dummy_input,), verbose=False)
                return flops
            else:
                # 手动计算
                config = self.model.config
                hidden_size = getattr(config, 'hidden_size', 4096)
                num_layers = getattr(config, 'num_hidden_layers', 32)
                intermediate_size = getattr(config, 'intermediate_size', 11008)
                
                # 自注意力的FLOPs: 4 * hidden_size * hidden_size * seq_len
                attention_flops = 4 * hidden_size * hidden_size * seq_len
                
                # FFN的FLOPs: 2 * hidden_size * intermediate_size * seq_len
                ffn_flops = 2 * hidden_size * intermediate_size * seq_len
                
                # 总FLOPs
                total_flops = num_layers * (attention_flops + ffn_flops)
                
                return total_flops
        except Exception as e:
            print(f"LLM FLOPs计算失败: {e}")
            # 回退到手动计算
            config = self.model.config
            hidden_size = getattr(config, 'hidden_size', 4096)
            num_layers = getattr(config, 'num_hidden_layers', 32)
            intermediate_size = getattr(config, 'intermediate_size', 11008)
            seq_len = input_ids.shape[1] + visual_tokens_count
            
            attention_flops = 4 * hidden_size * hidden_size * seq_len
            ffn_flops = 2 * hidden_size * intermediate_size * seq_len
            total_flops = num_layers * (attention_flops + ffn_flops)
            
            return total_flops
    
    def calculate_tokenizer_flops(self, text_input):
        """计算分词器FLOPs"""
        # 分词器的FLOPs相对较小，主要是字符串处理
        return len(text_input) * 100  # 简单估算
    
    def calculate_total_flops(self, image_tensor, text_input, visual_tokens_count):
        """计算总FLOPs"""
        print("计算编码器FLOPs...")
        if FVCORE_AVAILABLE:
            self.encoder_flops = self.calculate_encoder_flops_with_fvcore(image_tensor)
        elif THOP_AVAILABLE:
            self.encoder_flops = self.calculate_encoder_flops_with_thop(image_tensor)
        else:
            self.encoder_flops = self.calculate_encoder_flops_manual(image_tensor)
        
        print("计算连接器FLOPs...")
        # 创建虚拟的visual features
        dummy_visual_features = torch.randn(1, visual_tokens_count, 4096)
        self.connector_flops = self.calculate_connector_flops(dummy_visual_features)
        
        print("计算分词器FLOPs...")
        self.tokenizer_flops = self.calculate_tokenizer_flops(text_input)
        
        print("计算LLM FLOPs...")
        dummy_input_ids = torch.randint(0, 1000, (1, 10))
        self.llm_flops = self.calculate_llm_flops(dummy_input_ids, visual_tokens_count)
        
        # 总FLOPs
        total_flops = self.encoder_flops + self.connector_flops + self.tokenizer_flops + self.llm_flops
        
        return {
            'encoder_flops': self.encoder_flops,
            'connector_flops': self.connector_flops,
            'tokenizer_flops': self.tokenizer_flops,
            'llm_flops': self.llm_flops,
            'total_flops': total_flops
        }


def load_model_and_tokenizer(model_path, vision_encoder="fastvithd"):
    """加载模型和分词器"""
    disable_torch_init()
    
    # 获取模型名称
    model_name = get_model_name_from_path(model_path)
    
    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    if 'qwen' in model_name.lower():
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    # 加载图像处理器
    image_processor = model.get_vision_tower().image_processor
    
    return model, tokenizer, image_processor


def create_test_image(resolution=1024):
    """创建测试图像"""
    # 创建一个随机图像用于测试
    image = Image.fromarray(np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8))
    return image


def test_vlm_flops_advanced(args):
    """测试VLM模型FLOPs (高级版本)"""
    print("="*60)
    print("FastVLM FLOPs 计算测试 (高级版本)")
    print("="*60)
    
    # 检查FLOPs计算库
    if FVCORE_AVAILABLE:
        print("使用 fvcore 进行FLOPs计算")
    elif THOP_AVAILABLE:
        print("使用 thop 进行FLOPs计算")
    else:
        print("使用手动估算进行FLOPs计算")
    
    # 加载模型和分词器
    print(f"加载模型: {args.model_path}")
    model, tokenizer, image_processor = load_model_and_tokenizer(args.model_path, args.vision_encoder)
    
    # 创建FLOPs计算器
    flops_calculator = AdvancedFLOPsCalculator(model, tokenizer, image_processor, args.vision_encoder)
    
    # 创建测试图像
    print(f"创建测试图像: {args.resolution}x{args.resolution}")
    test_image = create_test_image(args.resolution)
    
    # 处理图像
    image_tensor = image_processor(test_image, return_tensors='pt')['pixel_values']
    
    # 计算visual tokens
    if args.visual_tokens is not None:
        visual_tokens_count = args.visual_tokens
    else:
        visual_tokens_count = calculate_visual_tokens(args.resolution, args.vision_encoder, args.patch_size)
    
    # 创建测试文本
    test_text = "Describe this image in detail."
    
    # 计算FLOPs
    print("开始计算FLOPs...")
    flops_results = flops_calculator.calculate_total_flops(image_tensor, test_text, visual_tokens_count)
    
    # 获取模型配置信息
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    llm_backbone = "Qwen2" if config.model_type == "llava_qwen2" else "Llama"
    
    # 设置模型配置名称
    if args.vision_encoder == "fastvithd":
        model_config = 'LLaVA_FastViTHD_7B_Stage2'
    elif args.vision_encoder == "clip":
        model_config = 'LLaVA_CLIP_ViT_L_14_336px'
    elif args.vision_encoder == "clip_s2":
        model_config = 'LLaVA_CLIP_ViT_L_14_336px_S2'
    elif args.vision_encoder == "clip_llava16":
        model_config = 'LLaVA_CLIP_ViT_L_14_336px_LLaVA16'
    else:
        model_config = f'LLaVA_{args.vision_encoder.upper()}'
    
    # 打印结果
    print("\n" + "="*60)
    print("FLOPs 计算结果")
    print("="*60)
    print(f"模型配置: {model_config}")
    print(f"视觉编码器: {args.vision_encoder}")
    print(f"LLM骨干: {llm_backbone}")
    print(f"输入分辨率: {args.resolution}x{args.resolution}")
    print(f"Visual Tokens: {visual_tokens_count}")
    print(f"模型路径: {args.model_path}")
    print("-"*60)
    print(f"编码器FLOPs: {flops_results['encoder_flops']:,.0f}")
    print(f"连接器FLOPs: {flops_results['connector_flops']:,.0f}")
    print(f"分词器FLOPs: {flops_results['tokenizer_flops']:,.0f}")
    print(f"LLM FLOPs: {flops_results['llm_flops']:,.0f}")
    print("-"*60)
    print(f"总FLOPs: {flops_results['total_flops']:,.0f}")
    print(f"总FLOPs (G): {flops_results['total_flops'] / 1e9:.2f}G")
    print(f"总FLOPs (T): {flops_results['total_flops'] / 1e12:.3f}T")
    print("="*60)
    
    # 计算各部分占比
    total = flops_results['total_flops']
    encoder_ratio = flops_results['encoder_flops'] / total * 100
    connector_ratio = flops_results['connector_flops'] / total * 100
    tokenizer_ratio = flops_results['tokenizer_flops'] / total * 100
    llm_ratio = flops_results['llm_flops'] / total * 100
    
    print(f"\n各部分FLOPs占比:")
    print(f"编码器: {encoder_ratio:.1f}%")
    print(f"连接器: {connector_ratio:.1f}%")
    print(f"分词器: {tokenizer_ratio:.1f}%")
    print(f"LLM: {llm_ratio:.1f}%")
    
    # 保存结果
    if args.output_file:
        results = {
            'model_config': model_config,
            'vision_encoder': args.vision_encoder,
            'llm': llm_backbone,
            'input_resolution': f'{args.resolution}x{args.resolution}',
            'visual_tokens': visual_tokens_count,
            'model_path': args.model_path,
            'encoder_flops': flops_results['encoder_flops'],
            'connector_flops': flops_results['connector_flops'],
            'tokenizer_flops': flops_results['tokenizer_flops'],
            'llm_flops': flops_results['llm_flops'],
            'total_flops': flops_results['total_flops'],
            'total_flops_g': flops_results['total_flops'] / 1e9,
            'total_flops_t': flops_results['total_flops'] / 1e12,
            'encoder_ratio': encoder_ratio,
            'connector_ratio': connector_ratio,
            'tokenizer_ratio': tokenizer_ratio,
            'llm_ratio': llm_ratio,
            'flops_library': 'fvcore' if FVCORE_AVAILABLE else ('thop' if THOP_AVAILABLE else 'manual'),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n结果已保存到: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM模型FLOPs计算 (高级版本)")
    parser.add_argument("--model-path", type=str, required=True, help="VLM模型路径")
    parser.add_argument("--resolution", type=int, default=1024, 
                       choices=[224, 256, 336, 384, 448, 512, 672, 768, 1008, 1024, 1152, 1344, 1536], 
                       help="输入分辨率")
    parser.add_argument("--output-file", type=str, default=None, help="输出结果文件")
    parser.add_argument("--vision-encoder", type=str, default="fastvithd", 
                       choices=["fastvithd", "clip", "clip_s2", "clip_llava16"],
                       help="视觉编码器类型")
    parser.add_argument("--visual-tokens", type=int, default=None, 
                       help="Visual token数量 (如果为None，将根据分辨率和编码器自动计算)")
    parser.add_argument("--patch-size", type=int, default=None,
                       help="Patch大小 (如果为None，将使用每个编码器的默认值)")
    
    args = parser.parse_args()
    
    if args.output_file is None:
        if args.visual_tokens is not None:
            visual_tokens_str = f"vt{args.visual_tokens}"
        else:
            visual_tokens = calculate_visual_tokens(args.resolution, args.vision_encoder, args.patch_size)
            visual_tokens_str = f"vt{visual_tokens}"
        
        args.output_file = f"flops_advanced_results_{args.vision_encoder}_{args.resolution}x{args.resolution}_{visual_tokens_str}.json"
    
    test_vlm_flops_advanced(args)
