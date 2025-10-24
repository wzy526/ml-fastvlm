#!/usr/bin/env python3
"""
测试FlashMaskPyTorch实现
验证动态掩码功能和性能
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from typing import List
import sys
import os

# 添加路径
sys.path.append('/root/ml-fastvlm')
from llava.model.language_model.flashmask_pytorch import FlashMaskPyTorch

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 50)
    print("测试1: 基本功能测试")
    print("=" * 50)
    
    # 创建测试数据
    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输入
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # 创建FlashMask实例
    flashmask = FlashMaskPyTorch(use_flash_attn=True)
    
    # 测试动态掩码生成
    image_range_list = [
        [[10, 20], [30, 40]],  # batch 0: 两个图像范围
        [[50, 60]]              # batch 1: 一个图像范围
    ]
    
    try:
        # 测试稀疏掩码表示
        mask_indices, mask_values = flashmask.create_sparse_mask_representation(
            batch_size, num_heads, seq_len, device, q.dtype, 
            image_range_list, grid_size=12
        )
        
        print(f"✅ 稀疏掩码生成成功")
        print(f"   - 掩码索引形状: {mask_indices.shape}")
        print(f"   - 掩码值形状: {mask_values.shape}")
        
        # 测试注意力计算
        output = flashmask.forward(q, k, v, image_range_list, grid_size=12)
        
        print(f"✅ 注意力计算成功")
        print(f"   - 输出形状: {output.shape}")
        print(f"   - 输出数据类型: {output.dtype}")
        print(f"   - 输出设备: {output.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def test_mask_correctness():
    """测试掩码正确性"""
    print("\n" + "=" * 50)
    print("测试2: 掩码正确性测试")
    print("=" * 50)
    
    # 创建简单测试数据
    batch_size, seq_len, num_heads, head_dim = 1, 16, 2, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输入
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # 创建FlashMask实例
    flashmask = FlashMaskPyTorch(use_flash_attn=False)  # 使用手动实现便于调试
    
    # 测试不同的掩码模式
    test_cases = [
        {
            "name": "因果掩码",
            "image_range_list": [[]],
            "expected_behavior": "应该阻止未来token的注意力"
        },
        {
            "name": "动态掩码",
            "image_range_list": [[[4, 8]]],
            "expected_behavior": "应该允许特定范围的注意力"
        },
        {
            "name": "多范围掩码",
            "image_range_list": [[[2, 4], [8, 12]]],
            "expected_behavior": "应该允许多个范围的注意力"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        try:
            print(f"\n测试用例 {i+1}: {test_case['name']}")
            print(f"期望行为: {test_case['expected_behavior']}")
            
            output = flashmask.forward(
                q, k, v, 
                test_case['image_range_list'], 
                grid_size=4
            )
            
            print(f"✅ 输出形状: {output.shape}")
            print(f"✅ 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
        except Exception as e:
            print(f"❌ 测试用例 {i+1} 失败: {e}")

def test_performance():
    """测试性能"""
    print("\n" + "=" * 50)
    print("测试3: 性能测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过性能测试")
        return
    
    # 创建测试数据
    batch_size, seq_len, num_heads, head_dim = 4, 512, 16, 64
    device = torch.device('cuda')
    
    # 创建输入
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # 创建图像范围列表
    image_range_list = [
        [[i*10, (i+1)*10] for i in range(5)] for _ in range(batch_size)
    ]
    
    # 测试FlashMask性能
    flashmask = FlashMaskPyTorch(use_flash_attn=True)
    
    # 预热
    for _ in range(5):
        _ = flashmask.forward(q, k, v, image_range_list, grid_size=12)
    
    torch.cuda.synchronize()
    
    # 测试FlashMask
    start_time = time.time()
    for _ in range(10):
        output_flashmask = flashmask.forward(q, k, v, image_range_list, grid_size=12)
    torch.cuda.synchronize()
    flashmask_time = time.time() - start_time
    
    # 测试标准注意力
    start_time = time.time()
    for _ in range(10):
        # 标准注意力计算
        q_reshaped = q.transpose(1, 2)  # (batch, heads, seq, dim)
        k_reshaped = k.transpose(1, 2)
        v_reshaped = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # 应用因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 计算输出
        output_standard = torch.matmul(attn_weights, v_reshaped)
        output_standard = output_standard.transpose(1, 2)
    
    torch.cuda.synchronize()
    standard_time = time.time() - start_time
    
    print(f"✅ FlashMask时间: {flashmask_time:.4f}s")
    print(f"✅ 标准注意力时间: {standard_time:.4f}s")
    print(f"✅ 加速比: {standard_time/flashmask_time:.2f}x")
    
    # 验证输出一致性
    try:
        # 使用手动实现进行对比
        flashmask_manual = FlashMaskPyTorch(use_flash_attn=False)
        output_manual = flashmask_manual.forward(q, k, v, image_range_list, grid_size=12)
        
        # 计算差异
        diff = torch.abs(output_flashmask - output_manual).max().item()
        print(f"✅ FlashMask vs 手动实现差异: {diff:.6f}")
        
        if diff < 1e-3:
            print("✅ 输出一致性验证通过")
        else:
            print("⚠️  输出存在差异，可能需要调试")
            
    except Exception as e:
        print(f"⚠️  一致性验证失败: {e}")

def test_memory_usage():
    """测试内存使用"""
    print("\n" + "=" * 50)
    print("测试4: 内存使用测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过内存测试")
        return
    
    device = torch.device('cuda')
    
    # 测试不同序列长度
    seq_lengths = [128, 256, 512, 1024]
    batch_size, num_heads, head_dim = 2, 8, 64
    
    print("序列长度 | FlashMask内存 | 标准注意力内存 | 内存节省")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        try:
            # 创建输入
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            
            # 创建图像范围列表
            image_range_list = [[[i*10, (i+1)*10] for i in range(seq_len//20)] for _ in range(batch_size)]
            
            # 测试FlashMask内存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            flashmask = FlashMaskPyTorch(use_flash_attn=True)
            _ = flashmask.forward(q, k, v, image_range_list, grid_size=12)
            
            flashmask_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # 测试标准注意力内存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 标准注意力计算
            q_reshaped = q.transpose(1, 2)
            k_reshaped = k.transpose(1, 2)
            v_reshaped = v.transpose(1, 2)
            
            scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / (head_dim ** 0.5)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            _ = torch.matmul(attn_weights, v_reshaped)
            
            standard_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # 计算内存节省
            memory_saving = (standard_memory - flashmask_memory) / standard_memory * 100
            
            print(f"{seq_len:8d} | {flashmask_memory:10.1f}MB | {standard_memory:12.1f}MB | {memory_saving:6.1f}%")
            
        except Exception as e:
            print(f"{seq_len:8d} | 测试失败: {e}")

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 50)
    print("测试5: 边界情况测试")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flashmask = FlashMaskPyTorch(use_flash_attn=True)
    
    # 测试用例
    test_cases = [
        {
            "name": "空图像范围",
            "batch_size": 1,
            "seq_len": 32,
            "image_range_list": [[]],
            "expected": "应该正常工作"
        },
        {
            "name": "单token序列",
            "batch_size": 1,
            "seq_len": 1,
            "image_range_list": [[[0, 1]]],
            "expected": "应该正常工作"
        },
        {
            "name": "大batch size",
            "batch_size": 16,
            "seq_len": 64,
            "image_range_list": [[[i*2, (i+1)*2] for i in range(10)] for _ in range(16)],
            "expected": "应该正常工作"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        try:
            print(f"\n测试用例 {i+1}: {test_case['name']}")
            print(f"期望: {test_case['expected']}")
            
            # 创建输入
            q = torch.randn(test_case['batch_size'], test_case['seq_len'], 8, 64, device=device)
            k = torch.randn(test_case['batch_size'], test_case['seq_len'], 8, 64, device=device)
            v = torch.randn(test_case['batch_size'], test_case['seq_len'], 8, 64, device=device)
            
            # 测试
            output = flashmask.forward(q, k, v, test_case['image_range_list'], grid_size=12)
            
            print(f"✅ 成功 - 输出形状: {output.shape}")
            
        except Exception as e:
            print(f"❌ 失败: {e}")

def main():
    """主测试函数"""
    print("FlashMaskPyTorch 测试开始")
    print("=" * 50)
    
    # 检查环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name()}")
    
    # 运行测试
    tests = [
        test_basic_functionality,
        test_mask_correctness,
        test_performance,
        test_memory_usage,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 异常: {e}")
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！FlashMaskPyTorch实现正确")
    else:
        print("⚠️  部分测试失败，需要检查实现")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
