#!/usr/bin/env python3
#
# DAT-LLaVA-1.5 TTFT和FLOPs综合测试脚本 (GQA数据集)
# 基于训练脚本 train_dat_llava1_5_v2.sh 的配置
# 使用GQA数据集进行测试，使用训练好的checkpoint和config
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
import subprocess
import sys

# 导入现有的测试模块
from ttft_test import (
    calculate_visual_tokens, TestDataset, setup_distributed, 
    load_model, measure_fastvlm_ttft, test_fastvlm_ttft
)
from flops_test import (
    FVCoreFLOPsCalculator, load_model_and_tokenizer, 
    load_gqa_sample, test_vlm_flops_fvcore
)

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model import LlavaLlamaForCausalLM, LlavaQwen2ForCausalLM
import transformers


def get_training_config():
    """从训练脚本中获取配置信息 - 使用GQA数据集"""
    config = {
        'model_name_or_path': '/home/zhuofan.xia/gsva_pretrains/llava-v1_5-7b',
        'vision_tower': '/home/zhuofan.xia/gsva_pretrains/clip-vit-large-patch14-336',
        'data_path': '/perception-hl/zhuofan.xia/data/gqa/questions/val_all_questions.json',  # GQA验证集
        'image_folder': '/perception-hl/zhuofan.xia/data/gqa/images',  # GQA图像文件夹
        'output_dir': '/perception-hl/zhuofan.xia/vlm_exps/textdat/tdat-7b-l0d32-s12g8z3',  
        'extra_yaml_path': './configs/llava1_5_v1.yaml',
        'mm_projector_type': 'mlp2x_gelu',
        'mm_vision_select_layer': -2,
        'mm_use_im_start_end': False,
        'mm_use_im_patch_token': False,
        'image_aspect_ratio': 'pad',
        'model_max_length': 2048,
        'resolution': 336,  # 从yaml配置中获取
        'vision_encoder': 'clip',  # 使用CLIP作为视觉编码器
        'llm_type': 'llama'  # 使用LLaMA作为LLM backbone
    }
    return config


def find_latest_checkpoint(output_dir):
    """查找最新的checkpoint"""
    checkpoint_dirs = []
    for item in os.listdir(output_dir):
        if item.startswith('checkpoint-'):
            checkpoint_dirs.append(item)
    
    if not checkpoint_dirs:
        print(f"在 {output_dir} 中未找到checkpoint")
        return None
    
    # 按checkpoint步数排序
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
    latest_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
    print(f"找到最新checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def run_ttft_test(model_path, data_path, image_folder, output_file=None, 
                  resolution=336, max_samples=1000, vision_encoder="clip"):
    """运行TTFT测试"""
    print("="*60)
    print("开始TTFT测试")
    print("="*60)
    
    # 构建TTFT测试命令
    cmd = [
        sys.executable, 'ttft_test.py',
        '--model-path', model_path,
        '--data-path', data_path,
        '--image-folder', image_folder,
        '--resolution', str(resolution),
        '--vision-encoder', vision_encoder,
        '--max-samples', str(max_samples)
    ]
    
    if output_file:
        cmd.extend(['--output-file', output_file])
    
    print(f"运行TTFT测试命令: {' '.join(cmd)}")
    
    try:
        # 设置正确的工作目录为用户的工作目录
        work_dir = "/home/zhuofan.xia/ml-fastvlm"
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=work_dir)
        print("TTFT测试完成")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"TTFT测试失败: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def run_flops_test(model_path, output_file=None, resolution=336, vision_encoder="clip"):
    """运行FLOPs测试"""
    print("="*60)
    print("开始FLOPs测试")
    print("="*60)
    
    # 构建FLOPs测试命令
    cmd = [
        sys.executable, 'flops_test.py',
        '--model-path', model_path,
        '--resolution', str(resolution),
        '--vision-encoder', vision_encoder
    ]
    
    if output_file:
        cmd.extend(['--output-file', output_file])
    
    print(f"运行FLOPs测试命令: {' '.join(cmd)}")
    
    try:
        # 设置正确的工作目录为用户的工作目录
        work_dir = "/home/zhuofan.xia/ml-fastvlm"
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=work_dir)
        print("FLOPs测试完成")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FLOPs测试失败: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def run_comprehensive_test(args):
    """运行综合测试"""
    print("="*80)
    print("DAT-LLaVA-1.5 综合测试 (TTFT + FLOPs) - GQA数据集")
    print("="*80)
    
    # 获取训练配置
    config = get_training_config()
    print(f"训练配置: {config}")
    
    # 查找checkpoint
    if args.checkpoint_path:
        model_path = args.checkpoint_path
    else:
        model_path = find_latest_checkpoint(config['output_dir'])
        if not model_path:
            print("错误: 未找到checkpoint，请指定 --checkpoint-path")
            return False
    
    print(f"使用模型路径: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    # 设置输出文件
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    ttft_output = f"ttft_results_dat_llava1_5_gqa_{timestamp}.json"
    flops_output = f"flops_results_dat_llava1_5_gqa_{timestamp}.json"
    
    # 运行TTFT测试
    print("\n" + "="*60)
    print("1. 运行TTFT测试")
    print("="*60)
    
    ttft_success = run_ttft_test(
        model_path=model_path,
        data_path=config['data_path'],
        image_folder=config['image_folder'],
        output_file=ttft_output,
        resolution=args.resolution,
        max_samples=args.max_samples,
        vision_encoder=config['vision_encoder']
    )
    
    # 运行FLOPs测试
    print("\n" + "="*60)
    print("2. 运行FLOPs测试")
    print("="*60)
    
    flops_success = run_flops_test(
        model_path=model_path,
        output_file=flops_output,
        resolution=args.resolution,
        vision_encoder=config['vision_encoder']
    )
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总 (GQA数据集)")
    print("="*80)
    print(f"数据集: GQA (Graph Question Answering)")
    print(f"数据路径: {config['data_path']}")
    print(f"图像文件夹: {config['image_folder']}")
    print(f"模型路径: {model_path}")
    print(f"分辨率: {args.resolution}x{args.resolution}")
    print(f"视觉编码器: {config['vision_encoder']}")
    print(f"LLM类型: {config['llm_type']}")
    print("-"*80)
    print(f"TTFT测试: {'✅ 成功' if ttft_success else '❌ 失败'}")
    print(f"FLOPs测试: {'✅ 成功' if flops_success else '❌ 失败'}")
    
    if ttft_success:
        print(f"TTFT结果文件: {ttft_output}")
        # 读取并显示TTFT结果
        try:
            with open(ttft_output, 'r') as f:
                ttft_results = json.load(f)
            print(f"平均TTFT: {ttft_results.get('avg_ttft_ms', 'N/A')}ms")
        except Exception as e:
            print(f"读取TTFT结果失败: {e}")
    
    if flops_success:
        print(f"FLOPs结果文件: {flops_output}")
        # 读取并显示FLOPs结果
        try:
            with open(flops_output, 'r') as f:
                flops_results = json.load(f)
            print(f"总FLOPs: {flops_results.get('total_flops', 'N/A'):,}")
            print(f"总FLOPs (G): {flops_results.get('total_flops_g', 'N/A'):.2f}G")
        except Exception as e:
            print(f"读取FLOPs结果失败: {e}")
    
    # 创建综合结果文件
    comprehensive_results = {
        'dataset_info': {
            'name': 'GQA (Graph Question Answering)',
            'data_path': config['data_path'],
            'image_folder': config['image_folder'],
            'description': 'GQA数据集用于测试视觉问答性能'
        },
        'model_path': model_path,
        'training_config': config,
        'test_config': {
            'resolution': args.resolution,
            'max_samples': args.max_samples,
            'vision_encoder': config['vision_encoder'],
            'llm_type': config['llm_type']
        },
        'test_results': {
            'ttft_success': ttft_success,
            'flops_success': flops_success,
            'ttft_output_file': ttft_output if ttft_success else None,
            'flops_output_file': flops_output if flops_success else None
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    comprehensive_output = f"comprehensive_test_results_dat_llava1_5_gqa_{timestamp}.json"
    with open(comprehensive_output, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n综合结果文件: {comprehensive_output}")
    
    return ttft_success and flops_success


def main():
    parser = argparse.ArgumentParser(description="DAT-LLaVA-1.5 综合测试 (TTFT + FLOPs) - GQA数据集")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                       help="训练好的checkpoint路径 (如果为None，将自动查找最新的checkpoint)")
    parser.add_argument("--resolution", type=int, default=336, 
                       choices=[224, 256, 336, 384, 448, 512, 672, 768, 1008, 1024, 1152, 1344, 1536],
                       help="输入分辨率")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="TTFT测试的最大样本数")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录 (如果为None，将使用当前目录)")
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # 运行综合测试
    success = run_comprehensive_test(args)
    
    if success:
        print("\n🎉 所有测试完成成功!")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
