#!/usr/bin/env bash
# 实验 pod 上的 system-level 推理基准 (最全版):
#   全部 5 个任务 x 7 档分辨率 x batch 到 32 x decode {128,512}
#   对照: Qwen 原生 / Base / DAT-separate / DAT-fused
# 模型: 预先用 scripts/make_dat_ckpt.py 生成的随机 DAT ckpt (测速与训练权重等价)。
#
# 用法:
#   bash scripts/run_sysbench_pod.sh 2>&1 | tee sysbench.log            # 3B 默认
#   MODEL_SIZE=7B bash scripts/run_sysbench_pod.sh 2>&1 | tee sys7b.log # 7B
#   MODEL_SIZE=7B TASKS=prefill bash scripts/run_sysbench_pod.sh        # 7B 只跑 prefill
set -euo pipefail

source /home/ea-cvfa-aigc-x2v-2/xzf/venvs/vldat/bin/activate
cd /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm

MODEL_SIZE="${MODEL_SIZE:-3B}"
BASE_MODEL="${BASE_MODEL:-/workspace/model_cache/Qwen2.5-VL-${MODEL_SIZE}-Instruct}"
DAT_CKPT="${DAT_CKPT:-${BASE_MODEL}-DAT-rand}"
TASKS="${TASKS:-all}"

if [[ ! -f "$DAT_CKPT/config.json" ]]; then
    echo "[ERROR] 缺 DAT ckpt: $DAT_CKPT" >&2
    echo "        先跑: python scripts/make_dat_ckpt.py --base-model $BASE_MODEL --output $DAT_CKPT" >&2
    exit 1
fi

echo "[cfg] base=$BASE_MODEL"
echo "[cfg] dat_ckpt=$DAT_CKPT  tasks=$TASKS"

# 分辨率须为 84 (= 28 x hr_scale=3) 的倍数
python test_inference_bench.py \
    --base-model "$BASE_MODEL" \
    --dat-ckpt "$DAT_CKPT" \
    --tasks $TASKS \
    --synthetic \
    --resolutions 672 1008 1344 2016 2688 3360 4032 \
    --batch-sizes 1 2 4 8 16 32 \
    --decode-lens 128 512 \
    --layerwise-r 1344 2688 \
    --warmup 2 --iters 3 \
    --tag "sys_${MODEL_SIZE}_$(date +%m%d_%H%M)"
