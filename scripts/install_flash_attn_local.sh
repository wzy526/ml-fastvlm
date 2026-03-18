#!/bin/bash
# 从本地 clone 的 flash-attention 安装到当前 conda 环境（建议先激活 fastvlm）
# 用法: conda activate fastvlm && bash scripts/install_flash_attn_local.sh

set -e
FLASH_SRC="${FLASH_ATTENTION_SRC:-/root/flash-attention}"
if [[ ! -d "$FLASH_SRC" ]]; then
  echo "Cloning flash-attention to $FLASH_SRC ..."
  git clone https://github.com/Dao-AILab/flash-attention.git "$FLASH_SRC"
  cd "$FLASH_SRC"
else
  cd "$FLASH_SRC"
  echo "Pulling latest in $FLASH_SRC ..."
  git pull origin main || true
fi

# 使用当前环境的 nvcc 所在前缀（conda 安装的 cuda 时）
NVCC=$(which nvcc 2>/dev/null || true)
if [[ -z "$NVCC" ]]; then
  echo "Error: nvcc not found. Activate conda env with cuda (e.g. conda activate fastvlm)."
  exit 1
fi
# CONDA_PREFIX 在 conda activate 后会自动设置
CONDA_PREFIX="${CONDA_PREFIX:-$(dirname "$(dirname "$(dirname "$NVCC")")")}"
# conda cuda toolkit 可能把 include 放在 targets/x86_64-linux/include
if [[ -f "$CONDA_PREFIX/targets/x86_64-linux/include/cuda.h" ]]; then
  export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
elif [[ -f "$CONDA_PREFIX/include/cuda.h" ]]; then
  export CUDA_HOME="$CONDA_PREFIX"
elif [[ -f "/usr/local/cuda/include/cuda.h" ]]; then
  export CUDA_HOME="/usr/local/cuda"
else
  echo "Warning: cuda.h not found. Set CUDA_HOME manually if build fails."
fi
export PATH="${CUDA_HOME:-$CUDA_PREFIX/bin}:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:-$CUDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "Using CUDA_HOME=${CUDA_HOME:-$CUDA_PREFIX}"
# conda 的 nvcc 会调用 cicc，需在 PATH 中
NVVM_BIN="${CONDA_PREFIX:-$CUDA_PREFIX}/nvvm/bin"
[[ -d "$NVVM_BIN" ]] && export PATH="$NVVM_BIN:$PATH"
pip install ninja packaging -q
echo "Building and installing flash-attn (this may take 5–15 min) ..."
pip install . --no-build-isolation -v

echo "Verifying:"
python -c "from flash_attn import flash_attn_func; print('flash_attn OK')"
