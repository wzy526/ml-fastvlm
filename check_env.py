#!/usr/bin/env python3
"""检查环境依赖"""

try:
    import torch
    print("✅ PyTorch已安装:", torch.__version__)
    print("✅ CUDA可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("✅ CUDA版本:", torch.version.cuda)
        print("✅ GPU数量:", torch.cuda.device_count())
except ImportError:
    print("❌ PyTorch未安装")

try:
    import flash_attn
    print("✅ Flash Attention已安装")
except ImportError:
    print("⚠️  Flash Attention未安装")

try:
    import einops
    print("✅ Einops已安装")
except ImportError:
    print("⚠️  Einops未安装")
