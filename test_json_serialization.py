#!/usr/bin/env python3
"""
测试脚本：验证ModelDATExtraArguments的JSON序列化修复
"""

import json
import sys
import os

# 添加项目路径
sys.path.append('/root/ml-fastvlm')

from llava.train.train_dat import ModelDATExtraArguments
from llava.model.language_model.llava_llama_dat import LlavaLlamaDATConfig

def test_dataclass_serialization():
    """测试ModelDATExtraArguments的序列化"""
    print("测试ModelDATExtraArguments序列化...")
    
    # 创建ModelDATExtraArguments实例
    dat_args = ModelDATExtraArguments(
        lr_image_size=256,
        hr_image_size=1008,
        grid_size=12,
        off_ksize=3,
        off_grps=8,
        inter_size=256,
        lr_size=24,
        hd_proj=True,
        layers=['D'] * 32,
        use_sdpa=False
    )
    
    # 测试to_dict方法
    try:
        dat_dict = dat_args.to_dict()
        print("✓ to_dict() 方法工作正常")
        print(f"  字典内容: {dat_dict}")
    except Exception as e:
        print(f"✗ to_dict() 方法失败: {e}")
        return False
    
    # 测试JSON序列化
    try:
        json_str = json.dumps(dat_dict)
        print("✓ JSON序列化成功")
        print(f"  JSON长度: {len(json_str)} 字符")
    except Exception as e:
        print(f"✗ JSON序列化失败: {e}")
        return False
    
    return True

def test_config_serialization():
    """测试LlavaLlamaDATConfig的序列化"""
    print("\n测试LlavaLlamaDATConfig序列化...")
    
    # 创建配置实例
    config = LlavaLlamaDATConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
    )
    
    # 添加dat_extra_args
    from llava.train.train_dat import ModelDATExtraArguments
    config.dat_extra_args = ModelDATExtraArguments()
    
    # 测试配置序列化
    try:
        config_dict = config.to_dict()
        print("✓ config.to_dict() 方法工作正常")
        print(f"  dat_extra_args类型: {type(config_dict.get('dat_extra_args'))}")
    except Exception as e:
        print(f"✗ config.to_dict() 方法失败: {e}")
        return False
    
    # 测试JSON序列化
    try:
        json_str = json.dumps(config_dict)
        print("✓ 配置JSON序列化成功")
        print(f"  JSON长度: {len(json_str)} 字符")
    except Exception as e:
        print(f"✗ 配置JSON序列化失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("开始测试JSON序列化修复...")
    print("=" * 50)
    
    success = True
    
    # 测试dataclass序列化
    if not test_dataclass_serialization():
        success = False
    
    # 测试配置序列化
    if not test_config_serialization():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 所有测试通过！JSON序列化问题已修复。")
        return 0
    else:
        print("✗ 部分测试失败，请检查修复。")
        return 1

if __name__ == "__main__":
    exit(main())
