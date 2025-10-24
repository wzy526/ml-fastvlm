#!/usr/bin/env python3
"""
测试简单的FlashMask实现
基于PyTorch原生优化
"""

import torch
import time
import sys
import os

# 添加路径
sys.path.append('/root/ml-fastvlm')
from llava.model.language_model.flashmask_simple import FlashMaskPyTorch, flashmask_attention

def test_simple_implementation():
    """测试简单实现"""
    print("=" * 50)
    print("测试简单FlashMask实现")
    print("=" * 50)
    
    # 创建测试数据
    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输入
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # 创建图像范围列表
    image_range_list = [
        [[10, 20], [30, 40]],  # batch 0
        [[50, 60]]              # batch 1
    ]
    
    print(f"✅ 输入形状: {q.shape}")
    print(f"✅ 设备: {device}")
    
    # 测试函数调用
    try:
        output = flashmask_attention(q, k, v, image_range_list, grid_size=12)
        print(f"✅ 函数调用成功")
        print(f"   - 输出形状: {output.shape}")
        print(f"   - 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    except Exception as e:
        print(f"❌ 函数调用失败: {e}")
        return False
    
    # 测试类调用
    try:
        flashmask = FlashMaskPyTorch(use_flash_attn=True)
        output_class = flashmask.forward(q, k, v, image_range_list, grid_size=12)
        print(f"✅ 类调用成功")
        print(f"   - 输出形状: {output_class.shape}")
        
        # 验证一致性
        diff = torch.abs(output - output_class).max().item()
        print(f"✅ 一致性检查: {diff:.6f}")
        
    except Exception as e:
        print(f"❌ 类调用失败: {e}")
        return False
    
    return True

def test_performance():
    """测试性能"""
    print("\n" + "=" * 50)
    print("性能测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过性能测试")
        return
    
    # 创建测试数据
    batch_size, seq_len, num_heads, head_dim = 4, 512, 16, 64
    device = torch.device('cuda')
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    
    # 创建图像范围列表
    image_range_list = [
        [[i*10, (i+1)*10] for i in range(5)] for _ in range(batch_size)
    ]
    
    # 预热
    for _ in range(5):
        _ = flashmask_attention(q, k, v, image_range_list, grid_size=12)
    
    torch.cuda.synchronize()
    
    # 测试FlashMask
    start_time = time.time()
    for _ in range(10):
        output_flashmask = flashmask_attention(q, k, v, image_range_list, grid_size=12)
    torch.cuda.synchronize()
    flashmask_time = time.time() - start_time
    
    # 测试标准注意力
    start_time = time.time()
    for _ in range(10):
        # 标准注意力计算
        q_reshaped = q.transpose(1, 2)  # (batch, heads, seq, dim)
        k_reshaped = k.transpose(1, 2)
        v_reshaped = v.transpose(1, 2)
        
        scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / (head_dim ** 0.5)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output_standard = torch.matmul(attn_weights, v_reshaped)
        output_standard = output_standard.transpose(1, 2)
    
    torch.cuda.synchronize()
    standard_time = time.time() - start_time
    
    print(f"✅ FlashMask时间: {flashmask_time:.4f}s")
    print(f"✅ 标准注意力时间: {standard_time:.4f}s")
    print(f"✅ 加速比: {standard_time/flashmask_time:.2f}x")
    
    # 验证输出一致性
    try:
        diff = torch.abs(output_flashmask - output_standard).max().item()
        print(f"✅ 输出差异: {diff:.6f}")
        
        if diff < 1e-3:
            print("✅ 输出一致性验证通过")
        else:
            print("⚠️  输出存在差异")
            
    except Exception as e:
        print(f"⚠️  一致性验证失败: {e}")

def test_memory_usage():
    """测试内存使用"""
    print("\n" + "=" * 50)
    print("内存使用测试")
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
            
            _ = flashmask_attention(q, k, v, image_range_list, grid_size=12)
            
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

def main():
    """主测试函数"""
    print("简单FlashMask实现测试")
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
        test_simple_implementation,
        test_performance,
        test_memory_usage
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
        print("🎉 所有测试通过！简单FlashMask实现正确")
    else:
        print("⚠️  部分测试失败，需要检查实现")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
