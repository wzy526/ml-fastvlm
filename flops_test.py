#!/usr/bin/env python3
#
# FastVLM FLOPs 计算脚本 - 使用fvcore
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
from torchvision import transforms

try:
    from fvcore.nn import FlopCountAnalysis, flop_count
    FVCORE_AVAILABLE = True
    print("✓ fvcore 已安装")
except ImportError:
    FVCORE_AVAILABLE = False
    print("✗ fvcore 未安装，请安装: pip install fvcore")

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
        # Vanilla LLaVA 会将输入图像缩放到 336 后送入 CLIP-L/14，视觉 token 固定为 (336/14)^2=576
        return 576
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
    elif vision_encoder == "custom":
        # 自定义编码器，需要用户指定visual token数量
        print("警告: 使用自定义视觉编码器，无法自动计算visual token数量")
        print("请使用 --visual-tokens 参数手动指定")
        if patch_size is None:
            patch_size = 14
        num_patches = (resolution // patch_size) ** 2
        return num_patches
    else:
        # 默认使用CLIP的方式计算
        if patch_size is None:
            patch_size = 14
        num_patches = (resolution // patch_size) ** 2
        return num_patches


class FVCoreFLOPsCalculator:
    """使用fvcore计算VLM模型FLOPs"""
    
    def __init__(self, model, tokenizer, image_processor, vision_encoder="fastvithd", llm_type="qwen2"):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.vision_encoder = vision_encoder
        self.llm_type = llm_type
        
        if not FVCORE_AVAILABLE:
            raise ImportError("fvcore未安装，请安装: pip install fvcore")
    
    def calculate_encoder_flops(self, image_tensor):
        """使用fvcore计算视觉编码器FLOPs"""
        try:
            vision_tower = self.model.get_vision_tower()
            
            print(f"计算视觉编码器FLOPs...")
            print(f"输入张量形状: {image_tensor.shape}")
            
            # 检查vision_tower是否已加载
            if hasattr(vision_tower, 'is_loaded') and not vision_tower.is_loaded:
                print("视觉编码器未加载，正在加载...")
                vision_tower.load_model()
            
            try:
                device = next(vision_tower.parameters()).device
                dtype = next(vision_tower.parameters()).dtype
            except StopIteration:
                # 如果vision_tower没有参数，使用模型的设备和数据类型
                device = next(self.model.parameters()).device
                dtype = next(self.model.parameters()).dtype
            image_tensor = image_tensor.to(device=device, dtype=dtype)
            
            # 使用FlopCountAnalysis
            flops_analyzer = FlopCountAnalysis(vision_tower, (image_tensor,))
            flops = flops_analyzer.total()
            
            print(f"视觉编码器FLOPs: {flops:,.0f}")
            
            return flops
        except Exception as e:
            print(f"视觉编码器FLOPs计算失败: {e}")
            return 0
    
    def calculate_connector_flops(self, visual_features):
        """使用fvcore计算连接器(投影层)FLOPs"""
        try:
            if hasattr(self.model, 'get_model') and hasattr(self.model.get_model(), 'mm_projector'):
                projector = self.model.get_model().mm_projector
                
                print(f"计算连接器FLOPs...")
                print(f"输入特征形状: {visual_features.shape}")

                device = next(projector.parameters()).device
                dtype = next(projector.parameters()).dtype
                visual_features = visual_features.to(device=device, dtype=dtype)
                
                # 使用FlopCountAnalysis
                flops_analyzer = FlopCountAnalysis(projector, (visual_features,))
                flops = flops_analyzer.total()
                
                print(f"连接器FLOPs: {flops:,.0f}")
                
                return flops
        except Exception as e:
            print(f"连接器FLOPs计算失败: {e}")
            return 0
    
    def calculate_llm_flops(self, input_ids, visual_tokens_count):
        """使用自定义fvcore计算LLM的FLOPs"""
        try:
            seq_len = input_ids.shape[1] + visual_tokens_count
            
            print(f"计算LLM FLOPs...")
            print(f"输入序列长度: {input_ids.shape[1]}")
            print(f"Visual tokens数量: {visual_tokens_count}")
            print(f"总序列长度: {seq_len}")
            
            # device and dtype
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype

            total_seq_len = visual_tokens_count + input_ids.shape[1]
            dummy_input_ids = torch.randint(0, 1000, (1, total_seq_len), device=device, dtype=torch.long)
            
            # 方法1: 使用fvcore计算 (关闭dynamic cache)
            print("="*40)
            print("方法1: 使用fvcore计算LLM FLOPs")
            print("="*40)
            
            # dynamic cache
            original_use_cache = self.model.config.use_cache
            self.model.config.use_cache = False
            print(f"关闭dynamic cache: use_cache = {self.model.config.use_cache}")
            
            try:
                # 使用FlopCountAnalysis计算LLM FLOPs
                flops_analyzer = FlopCountAnalysis(self.model, (dummy_input_ids,))
                fvcore_flops = flops_analyzer.total()
                print(f"fvcore计算的LLM FLOPs: {fvcore_flops:,.0f}")
                fvcore_success = True
            except Exception as e:
                print(f"fvcore计算LLM FLOPs失败: {e}")
                fvcore_flops = 0
                fvcore_success = False
            finally:
                self.model.config.use_cache = original_use_cache
                print(f"恢复dynamic cache设置: use_cache = {self.model.config.use_cache}")
            
            # 方法2: 手动计算
            print("="*40)
            print("方法2: 手动计算LLM FLOPs")
            print("="*40)
            
            config = self.model.config
            hidden_size = getattr(config, 'hidden_size', 3584)  # Qwen2-7B的hidden_size
            num_layers = getattr(config, 'num_hidden_layers', 28)  # Qwen2-7B的层数
            intermediate_size = getattr(config, 'intermediate_size', 18944)  # Qwen2-7B的intermediate_size
            num_attention_heads = getattr(config, 'num_attention_heads', 28)  # 注意力头数量
            num_key_value_heads = getattr(config, 'num_key_value_heads', 4)  # KV头数量 (GQA)
            
            # head dim
            head_dim = hidden_size // num_attention_heads  # 128
            kv_head_dim = (hidden_size // num_attention_heads) * num_key_value_heads  # 512
            
            # GQA self attention flops calculation
            # q_proj: hidden_size × hidden_size × seq_len
            q_flops = hidden_size * hidden_size * total_seq_len
            # k_proj: hidden_size × kv_head_dim × seq_len (GQA)
            k_flops = hidden_size * kv_head_dim * total_seq_len
            # v_proj: hidden_size × kv_head_dim × seq_len (GQA)
            v_flops = hidden_size * kv_head_dim * total_seq_len
            # attention: seq_len × seq_len × head_dim (per head) - QK + AV
            attn_flops = total_seq_len * total_seq_len * head_dim * num_attention_heads * 2
            # o_proj: hidden_size × hidden_size × seq_len
            o_flops = hidden_size * hidden_size * total_seq_len
            
            # total attention flops
            attention_flops = q_flops + k_flops + v_flops + attn_flops + o_flops
            
            # SwiGLU FFN
            #gate_proj: hidden_size × intermediate_size × seq_len
            gate_flops = hidden_size * intermediate_size * total_seq_len
            #up_proj: hidden_size × intermediate_size × seq_len
            up_flops = hidden_size * intermediate_size * total_seq_len
            # gate activation and gate multiplication: intermediate_size × seq_len
            gate_activation_flops = intermediate_size * total_seq_len
            #down_proj: intermediate_size × hidden_size × seq_len
            down_flops = intermediate_size * hidden_size * total_seq_len
            # total ffn flops
            ffn_flops = gate_flops + up_flops + gate_activation_flops + down_flops
            
            # norm layer flops (RMSNorm)
            # input_layernorm: hidden_size × seq_len
            input_layernorm_flops = hidden_size * total_seq_len
            # post_attention_layernorm: hidden_size × seq_len  
            post_attention_layernorm_flops = hidden_size * total_seq_len
            
            # each layer flops (include norm layer)
            layer_flops = attention_flops + ffn_flops + input_layernorm_flops + post_attention_layernorm_flops
            
            # embedding flops: vocab_size × hidden_size × seq_len
            vocab_size = getattr(config, 'vocab_size', 152064) # gqa vocab size
            # embedding_flops = vocab_size * hidden_size * total_seq_len
            
            # lm head flops: hidden_size × vocab_size × seq_len
            lm_head_flops = hidden_size * vocab_size * total_seq_len
            
            # norm layer flops (RMSNorm)
            final_norm_flops = hidden_size * total_seq_len
            
            # total flops (all layers + embedding + lm_head + final_norm)
            manual_flops = num_layers * layer_flops + lm_head_flops + final_norm_flops
            
            print(f"手动计算的LLM FLOPs: {manual_flops:,.0f}")
            
            # 比较结果
            print("="*40)
            print("FLOPs计算结果比较")
            print("="*40)
            print(f"fvcore计算: {fvcore_flops:,.0f}")
            print(f"手动计算: {manual_flops:,.0f}")
            
            if fvcore_success and fvcore_flops > 0:
                diff = abs(fvcore_flops - manual_flops)
                diff_percent = (diff / manual_flops) * 100
                print(f"差异: {diff:,.0f} ({diff_percent:.2f}%)")
                
                if diff_percent < 5:
                    print("✅ 两种方法结果基本一致 (差异 < 5%)")
                elif diff_percent < 20:
                    print("⚠️ 两种方法结果有一定差异 (差异 < 20%)")
                else:
                    print("❌ 两种方法结果差异较大 (差异 >= 20%)")
            else:
                print("❌ fvcore计算失败，无法比较")
            
            # 返回手动计算结果作为最终结果
            return manual_flops
            
        except Exception as e:
            print(f"LLM FLOPs calculation failed: {e}")
            return 0
    
    def calculate_total_flops(self, image_tensor, text_input, visual_tokens_count, text_tokens_count, tokenizer, extra_hr_kv_flops: int = 0):
        """计算整体 FLOPs（编码器 + 连接器 + LLM + 可选 DAT HR 额外 KV）"""
        print("="*60)
        print("使用fvcore计算FLOPs")
        print("="*60)

        encoder_flops = self.calculate_encoder_flops(image_tensor)

        print("获取真实视觉特征...")
        with torch.no_grad():
            vision_tower = self.model.get_vision_tower()
            if hasattr(vision_tower, 'is_loaded') and not vision_tower.is_loaded:
                print("视觉编码器未加载，正在加载...")
                vision_tower.load_model()
            try:
                device = next(vision_tower.parameters()).device
                dtype = next(vision_tower.parameters()).dtype
            except StopIteration:
                device = next(self.model.parameters()).device
                dtype = next(self.model.parameters()).dtype
            image_tensor_device = image_tensor.to(device=device, dtype=dtype)
            visual_features = vision_tower(image_tensor_device)

        connector_flops = self.calculate_connector_flops(visual_features)

        input_ids = tokenizer(text_input, return_tensors='pt')['input_ids']
        llm_flops = self.calculate_llm_flops(input_ids, visual_tokens_count)

        total_flops = encoder_flops + connector_flops + llm_flops + max(0, int(extra_hr_kv_flops))

        return {
            'encoder_flops': encoder_flops,
            'connector_flops': connector_flops,
            'llm_flops': llm_flops,
            'dat_hr_kv_extra_flops': int(extra_hr_kv_flops),
            'total_flops': total_flops
        }
    
def calculate_total_flops(self, image_tensor, text_input, visual_tokens_count, text_tokens_count, tokenizer, extra_hr_kv_flops: int = 0):
        """计算总FLOPs"""
        print("="*60)
        print("使用fvcore计算FLOPs")
        print("="*60)
        
        # 计算各部分FLOPs
        encoder_flops = self.calculate_encoder_flops(image_tensor)
        
        print("获取真实视觉特征...")
        with torch.no_grad():
            vision_tower = self.model.get_vision_tower()
            if hasattr(vision_tower, 'is_loaded') and not vision_tower.is_loaded:
                print("视觉编码器未加载，正在加载...")
                vision_tower.load_model()
            try:
                device = next(vision_tower.parameters()).device
                dtype = next(vision_tower.parameters()).dtype
            except StopIteration:
                # 如果vision_tower没有参数，使用模型的设备和数据类型
                device = next(self.model.parameters()).device
                dtype = next(self.model.parameters()).dtype
            image_tensor_device = image_tensor.to(device=device, dtype=dtype)
            visual_features = vision_tower(image_tensor_device)
        
        connector_flops = self.calculate_connector_flops(visual_features)
        

        input_ids = tokenizer(text_input, return_tensors='pt')['input_ids']
        llm_flops = self.calculate_llm_flops(input_ids, visual_tokens_count)
        
        # 总FLOPs (不包含tokenizer)
        total_flops = encoder_flops + connector_flops + llm_flops + max(0, int(extra_hr_kv_flops))
        
        return {
            'encoder_flops': encoder_flops,
            'connector_flops': connector_flops,
            'llm_flops': llm_flops,
            'dat_hr_kv_extra_flops': int(extra_hr_kv_flops),
            'total_flops': total_flops
        }


def load_model_and_tokenizer(model_path, vision_encoder="fastvithd", llm_type="auto", 
                           encoder_path=None, llm_path=None):
    """加载模型和分词器
    
    Args:
        model_path: 主模型路径
        vision_encoder: 视觉编码器类型
        llm_type: LLM类型 (auto/qwen2/llama/custom)
        encoder_path: 自定义视觉编码器路径
        llm_path: 自定义LLM路径
    """
    disable_torch_init()
    
    model_name = get_model_name_from_path(model_path)
    
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # determine llm type
    if llm_type == "auto":
        if 'qwen' in model_name.lower():
            actual_llm_type = "qwen2"
        else:
            actual_llm_type = "llama"
    else:
        actual_llm_type = llm_type
    
    print(f"使用LLM类型: {actual_llm_type}")
    
    # 修复配置中的子配置（与 ttft_test 保持一致，安全 no-op）
    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    from transformers import PretrainedConfig
    # decoder_config
    if hasattr(config, 'decoder_config') and isinstance(getattr(config, 'decoder_config'), dict):
        print("修复 decoder_config 配置...")
        config.decoder_config = PretrainedConfig.from_dict(config.decoder_config)
    # text_config（在你的权重里是 dict）
    if hasattr(config, 'text_config') and isinstance(getattr(config, 'text_config'), dict):
        print("修复 text_config 配置...")
        config.text_config = PretrainedConfig.from_dict(config.text_config)
    # vision_config（防御性）
    if hasattr(config, 'vision_config') and isinstance(getattr(config, 'vision_config'), dict):
        print("修复 vision_config 配置...")
        config.vision_config = PretrainedConfig.from_dict(config.vision_config)
    # generation_config
    if hasattr(config, 'generation_config') and isinstance(getattr(config, 'generation_config'), dict):
        from transformers import GenerationConfig
        print("修复 generation_config 配置...")
        config.generation_config = GenerationConfig.from_dict(config.generation_config)

    # 旧版 LLaVA 缺失的新字段，补默认值以兼容 transformers>=4.45
    missing_defaults = {
        'attention_dropout': 0.0,
        'hidden_dropout': 0.0,
        'attention_probs_dropout_prob': 0.0,
        'attention_bias': False,
        'mlp_bias': False,
        'use_cache': True,
        'rope_theta': 10000.0,
        'rope_scaling': None,
        'max_position_embeddings': getattr(config, 'max_position_embeddings', 4096),
        'rms_norm_eps': 1e-6,
        'initializer_range': 0.02,
        'use_sliding_window': False,
        'sliding_window': None,
        'max_window_layers': None,
        'tie_word_embeddings': False
    }
    for k, v in missing_defaults.items():
        if not hasattr(config, k):
            setattr(config, k, v)
    
    # 加载模型
    if actual_llm_type == "qwen2":
        model = LlavaQwen2ForCausalLM.from_pretrained(
            model_path,
            config=config,  # 传递修复后的配置
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    elif actual_llm_type == "llama":
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            config=config,  # 传递修复后的配置
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    elif actual_llm_type == "custom":
        if llm_path is None:
            raise ValueError("当llm_type为custom时，必须提供llm_path参数")

        model = LlavaQwen2ForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        raise ValueError(f"不支持的LLM类型: {actual_llm_type}")

    # 处理自定义视觉编码器
    if vision_encoder == "custom":
        if encoder_path is None:
            raise ValueError("当vision_encoder为custom时，必须提供encoder_path参数")
        print(f"使用自定义视觉编码器: {encoder_path}")

        vision_tower = model.get_vision_tower()
        if hasattr(vision_tower, 'image_processor'):
            image_processor = vision_tower.image_processor
        else:
            # CLIPVisionTower，用自带的processor
            image_processor = vision_tower.image_processor if hasattr(vision_tower, 'image_processor') else None
    else:
        print(f"使用视觉编码器: {vision_encoder}")
        vision_tower = model.get_vision_tower()
        if hasattr(vision_tower, 'image_processor'):
            image_processor = vision_tower.image_processor
        else:
            try:
                image_processor = vision_tower.image_processor
            except AttributeError:
                from transformers import CLIPImageProcessor
                # use local clip-vit-large-patch14-336
                image_processor = CLIPImageProcessor.from_pretrained("/data/gsva_pretrains/clip-vit-large-patch14-336")
    
    return model, tokenizer, image_processor, actual_llm_type


def load_gqa_sample(data_path, image_folder, resolution=1024):
    """加载GQA数据集中的一个样本"""
    import json

    with open(data_path, 'r') as f:
        data = json.load(f)

    sample_id = list(data.keys())[0]
    sample = data[sample_id]
    
    # load image
    image_path = os.path.join(image_folder, f"{sample['imageId']}.jpg")
    try:
        image = Image.open(image_path).convert('RGB')
        
        # resize
        w, h = image.size
        if w > h:
            new_w = resolution
            new_h = int(h * resolution / w)
        else:
            new_h = resolution
            new_w = int(w * resolution / h)
        
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 如果图像不是正方形，进行填充
        if new_w != resolution or new_h != resolution:
            background = Image.new('RGB', (resolution, resolution), (255, 255, 255))
            x = (resolution - new_w) // 2
            y = (resolution - new_h) // 2
            background.paste(image, (x, y))
            image = background
        
        return image, sample['question']
    except Exception as e:
        print(f"加载GQA样本失败: {e}")
        
        image = Image.fromarray(np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8))
    return image, "Describe this image in detail."


def test_vlm_flops_fvcore(args):
    """使用fvcore测试VLM模型FLOPs"""
    print("="*60)
    print("FastVLM FLOPs 计算测试 (使用fvcore)")
    print("="*60)
    
    if not FVCORE_AVAILABLE:
        print("错误: fvcore未安装，请安装: pip install fvcore")
        return
    
    if args.vision_encoder == "custom" and args.encoder_path is None:
        print("错误: 当vision_encoder为custom时，必须提供encoder_path参数")
        return
    
    if args.llm_type == "custom" and args.llm_path is None:
        print("错误: 当llm_type为custom时，必须提供llm_path参数")
        return
    
    # load model
    print(f"加载模型: {args.model_path}")
    if args.vision_encoder == "custom":
        print(f"自定义视觉编码器路径: {args.encoder_path}")
    if args.llm_type == "custom":
        print(f"自定义LLM路径: {args.llm_path}")
    
    model, tokenizer, image_processor, actual_llm_type = load_model_and_tokenizer(
        args.model_path, 
        args.vision_encoder, 
        args.llm_type,
        args.encoder_path,
        args.llm_path
    )
    
    flops_calculator = FVCoreFLOPsCalculator(
        model, 
        tokenizer, 
        image_processor, 
        args.vision_encoder,
        actual_llm_type
    )
    
    # load gqa
    data_path = "/data/gqa/testdev_balanced_questions.json"
    image_folder = "/data/gqa/images"

    
    print(f"加载GQA数据样本...")
    test_image, test_text = load_gqa_sample(data_path, image_folder, args.resolution)

    use_hr = getattr(args, 'use_raw_image', False) and args.resolution > 336
    dat_cfg = getattr(model.config, 'dat_extra_args', None)
    dat_hr_kv_extra_flops = 0
    if use_hr and dat_cfg is not None:
        # 将 PIL 图像转 tensor 并裁切为 zoom_ratio^2 个 336 子图
        to_tensor = transforms.ToTensor()
        raw = to_tensor(test_image).unsqueeze(0)  # [1,3,H,W] in [0,1]
        hr_image_size = int(dat_cfg['hr_image_size'])
        lr_image_size = int(dat_cfg['lr_image_size'])
        zoom_ratio = int(hr_image_size // lr_image_size)
        sub_size = lr_image_size
        # 保证尺寸匹配
        if raw.shape[-1] != hr_image_size:
            raw = torch.nn.functional.interpolate(raw, size=(hr_image_size, hr_image_size), mode='bilinear', align_corners=False)
        # 切分为子图
        patches = []
        for i in range(0, hr_image_size, sub_size):
            for j in range(0, hr_image_size, sub_size):
                patches.append(raw[:, :, i:i+sub_size, j:j+sub_size])
        image_tensor = torch.cat(patches, dim=0).half()  # [zoom^2,3,336,336]
        # HR额外开销单独算，LR还是走336
        visual_tokens_count = 576
        try:
            grid_size = int(dat_cfg.get('grid_size', 12))
            S = grid_size * grid_size
            hidden_size = int(getattr(model.config, 'hidden_size', 4096))
            num_layers = int(getattr(model.config, 'num_hidden_layers', 32))
            # 每层对 S 个 HR 采样 token 做 K/V 投影（2 * hidden_size^2 * S）
            dat_hr_kv_extra_flops = 2 * (hidden_size * hidden_size) * S * num_layers
        except Exception:
            dat_hr_kv_extra_flops = 0
    else:
        # 常规路径：单图由 processor 处理
        image_tensor = image_processor(test_image, return_tensors='pt')['pixel_values']
        if args.visual_tokens is not None:
            visual_tokens_count = args.visual_tokens
        else:
            visual_tokens_count = calculate_visual_tokens(args.resolution, args.vision_encoder, args.patch_size)
    
    # text tokens
    text_tokens = tokenizer(test_text, return_tensors='pt')['input_ids']
    text_tokens_count = text_tokens.shape[1]
    print(f"实际文本token数量: {text_tokens_count}")
    print(f"文本内容: {test_text}")
    
    # 若 DAT HR 生效，则把注意力对新增 HR-KV 的交互成本合入（prefill 口径：Nq×(Nkv+Nkv_hd)）
    if use_hr and dat_cfg is not None:
        try:
            grid_size = int(dat_cfg.get('grid_size', 12))
            S = grid_size * grid_size  # N_kv_hd
            hidden_size = int(getattr(model.config, 'hidden_size', 4096))
            num_layers = int(getattr(model.config, 'num_hidden_layers', 32))
            num_heads = int(getattr(model.config, 'num_attention_heads', 32))
            head_dim = hidden_size // max(1, num_heads)
            Nq_full = 576 + int(text_tokens_count)
            delta_attn = 2 * Nq_full * S * head_dim * num_heads * num_layers
            dat_hr_kv_extra_flops += int(delta_attn)
        except Exception:
            pass
    
    # flops
    print("开始计算FLOPs...")
    flops_results = flops_calculator.calculate_total_flops(
        image_tensor, test_text, visual_tokens_count, text_tokens_count, tokenizer,
        extra_hr_kv_flops=dat_hr_kv_extra_flops
    )
    
    # model config
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    llm_backbone = actual_llm_type.upper()
    
    # 根据encoder和LLM类型生成模型配置名称
    if args.vision_encoder == "fastvithd":
        encoder_name = "FastViTHD"
    elif args.vision_encoder == "clip":
        encoder_name = "CLIP_ViT_L_14_336px"
    elif args.vision_encoder == "clip_s2":
        encoder_name = "CLIP_ViT_L_14_336px_S2"
    elif args.vision_encoder == "clip_llava16":
        encoder_name = "CLIP_ViT_L_14_336px_LLaVA16"
    elif args.vision_encoder == "custom":
        encoder_name = "Custom"
    else:
        encoder_name = args.vision_encoder.upper()
    
    model_config = f'LLaVA_{encoder_name}_{llm_backbone}'

    print("\n" + "="*60)
    print("FLOPs 计算结果 (fvcore)")
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
    print(f"LLM FLOPs: {flops_results['llm_flops']:,.0f}")
    print("-"*60)
    print(f"总FLOPs: {flops_results['total_flops']:,.0f}")
    print(f"总FLOPs (G): {flops_results['total_flops'] / 1e9:.2f}G")
    print(f"总FLOPs (T): {flops_results['total_flops'] / 1e12:.3f}T")
    print("="*60)
    
    total = flops_results['total_flops']
    if total > 0:
        encoder_ratio = flops_results['encoder_flops'] / total * 100
        connector_ratio = flops_results['connector_flops'] / total * 100
        llm_ratio = flops_results['llm_flops'] / total * 100
        
        print(f"\n各部分FLOPs占比:")
        print(f"编码器: {encoder_ratio:.1f}%")
        print(f"连接器: {connector_ratio:.1f}%")
        print(f"LLM: {llm_ratio:.1f}%")
    
    if args.output_file:
        results = {
            'model_config': model_config,
            'vision_encoder': args.vision_encoder,
            'llm_type': actual_llm_type,
            'llm': llm_backbone,
            'input_resolution': f'{args.resolution}x{args.resolution}',
            'visual_tokens': visual_tokens_count,
            'model_path': args.model_path,
            'encoder_path': args.encoder_path,
            'llm_path': args.llm_path,
            'encoder_flops': flops_results['encoder_flops'],
            'connector_flops': flops_results['connector_flops'],
            'llm_flops': flops_results['llm_flops'],
            'dat_hr_kv_extra_flops': flops_results.get('dat_hr_kv_extra_flops', 0),
            'total_flops': flops_results['total_flops'],
            'total_flops_g': flops_results['total_flops'] / 1e9,
            'total_flops_t': flops_results['total_flops'] / 1e12,
            'encoder_ratio': encoder_ratio if total > 0 else 0,
            'connector_ratio': connector_ratio if total > 0 else 0,
            'llm_ratio': llm_ratio if total > 0 else 0,
            'flops_library': 'fvcore',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n结果已保存到: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM模型FLOPs计算 (使用fvcore)")
    parser.add_argument("--model-path", type=str, required=True, help="VLM模型路径")
    parser.add_argument("--resolution", type=int, default=1024, 
                       choices=[224, 256, 336, 384, 448, 512, 672, 768, 1008, 1024, 1152, 1344, 1536], 
                       help="输入分辨率")
    parser.add_argument("--output-file", type=str, default=None, help="输出结果文件")
    parser.add_argument("--vision-encoder", type=str, default="fastvithd", 
                       choices=["fastvithd", "clip", "clip_s2", "clip_llava16", "custom"],
                       help="视觉编码器类型")
    parser.add_argument("--llm-type", type=str, default="auto", 
                       choices=["auto", "qwen2", "llama", "custom"],
                       help="LLM类型 (auto: 自动检测, qwen2: Qwen2, llama: LLaMA, custom: 自定义)")
    parser.add_argument("--encoder-path", type=str, default=None,
                       help="自定义视觉编码器路径 (当vision-encoder为custom时使用)")
    parser.add_argument("--llm-path", type=str, default=None,
                       help="自定义LLM路径 (当llm-type为custom时使用)")
    parser.add_argument("--visual-tokens", type=int, default=None, 
                       help="Visual token数量 (如果为None，将根据分辨率和编码器自动计算)")
    parser.add_argument("--patch-size", type=int, default=None,
                       help="Patch大小 (如果为None，将使用每个编码器的默认值)")
    parser.add_argument("--use-raw-image", action="store_true", help="以原图张量切分子图进入 DAT HR 路径，计算 HR 下 FLOPs")
    
    args = parser.parse_args()
    
    if args.output_file is None:
        if args.visual_tokens is not None:
            visual_tokens_str = f"vt{args.visual_tokens}"
        else:
            visual_tokens = calculate_visual_tokens(args.resolution, args.vision_encoder, args.patch_size)
            visual_tokens_str = f"vt{visual_tokens}"
        
        args.output_file = f"flops_fvcore_results_{args.vision_encoder}_{args.resolution}x{args.resolution}_{visual_tokens_str}.json"
    
    test_vlm_flops_fvcore(args)
 