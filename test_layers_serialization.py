#!/usr/bin/env python3
"""
测试layers字段的JSON序列化
"""

import json
import sys
sys.path.append('/root/ml-fastvlm')

from llava.train.train_dat import ModelDATExtraArguments

def test_layers_serialization():
    """测试layers字段的序列化"""
    print("测试layers字段序列化...")
    
    # 创建ModelDATExtraArguments实例
    dat_args = ModelDATExtraArguments()
    
    print(f"layers类型: {type(dat_args.layers)}")
    print(f"layers内容: {dat_args.layers}")
    print(f"layers长度: {len(dat_args.layers)}")
    
    # 测试to_dict方法
    try:
        dat_dict = dat_args.to_dict()
        layers_from_dict = dat_dict['layers']
        print(f"✓ to_dict() 成功，layers: {layers_from_dict}")
        print(f"  类型: {type(layers_from_dict)}")
    except Exception as e:
        print(f"✗ to_dict() 失败: {e}")
        return False
    
    # 测试JSON序列化
    try:
        json_str = json.dumps(dat_dict)
        print(f"✓ JSON序列化成功")
        print(f"  JSON中的layers: {json.loads(json_str)['layers']}")
    except Exception as e:
        print(f"✗ JSON序列化失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_layers_serialization()
