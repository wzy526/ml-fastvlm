#!/usr/bin/env bash
# Pod base env 已带齐 torch/flash-attn/transformers 等重型依赖。
# 只建一个继承 base 的 venv, 补装小包 + 本 repo (--no-deps 防止旧钉子降级 base 的包)。
set -euo pipefail

XZF_ROOT="/home/ea-cvfa-aigc-x2v-2/xzf"
VENV="$XZF_ROOT/venvs/vldat"

/usr/local/miniconda3/bin/python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"

pip install wandb einops-exts markdown2
pip install --no-deps -e "$XZF_ROOT/ml-fastvlm"

python -c "
import torch, transformers, flash_attn
from llava.model.language_model import modeling_qwen2_5vl_dat
print(f'torch {torch.__version__} | transformers {transformers.__version__} | flash_attn {flash_attn.__version__} | cuda={torch.cuda.is_available()}')
print('DAT import OK')
"
echo "[DONE] 使用: source $VENV/bin/activate"
