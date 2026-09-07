#!/usr/bin/env bash
# Qwen3.5 DAT 的 system-level 推理测速: 原生 Qwen / Base / DAT-sep / DAT-fused,
# 外加 HD 早退 k 扫描。对应 0903 那批精度评测, 补上论文缺的效率侧数字。
#
# 早退的耗时只由 hd_early_exit_k 决定 (临时截断 visual.blocks), 与权重无关,
# 所以这里用一份 DAT ckpt 扫 k, 不必分别加载 ee12 / ee18 的 merged ckpt。
#
# 用法:
#   bash scripts/run_sysbench_qwen35.sh 2>&1 | tee sysbench_q35_4b.log
#   SIZE=2B EARLY_KS=0 bash scripts/run_sysbench_qwen35.sh
#   SIZE=4B TASKS=prefill EARLY_KS="0 12" bash scripts/run_sysbench_qwen35.sh
set -euo pipefail

CFS=/home/ea-cvfa-aigc-x2v-2/xzf
source "$CFS/venvs/vldat/bin/activate"
cd "$CFS/ml-fastvlm"

# pod 无外网
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_CACHE=/workspace/model_cache
# 日志重定向到文件时 stdout 默认块缓冲, 进度看不到实时状态
export PYTHONUNBUFFERED=1

SIZE="${SIZE:-4B}"
# 单卡测速: 多卡只会让 auto_burn 的干扰面变大
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 本地盘优先 (CFS 读模型慢), 回退到 CFS
pick_path() {
    for p in "$@"; do
        [[ -f "$p/config.json" ]] && { echo "$p"; return; }
    done
    return 1
}

case "$SIZE" in
    4B)
        BASE_MODEL=$(pick_path \
            /workspace/model_cache/Qwen3.5-4B \
            "$CFS/models/Qwen3.5-4B") || { echo "[ERROR] 找不到 Qwen3.5-4B" >&2; exit 1; }
        DAT_CKPT=$(pick_path \
            /workspace/model_cache/0901_sft_qwen35_4b_dat_ivcap-merged \
            "$CFS/vldat_experiments/0901_sft_qwen35_4b_dat_ivcap-merged") \
            || { echo "[ERROR] 找不到 4B DAT merged ckpt" >&2; exit 1; }
        DEFAULT_KS="0 12 18"
        ;;
    2B)
        BASE_MODEL=$(pick_path \
            /workspace/model_cache/Qwen3.5-2B \
            "$CFS/models/Qwen3.5-2B") || { echo "[ERROR] 找不到 Qwen3.5-2B" >&2; exit 1; }
        DAT_CKPT=$(pick_path \
            /workspace/model_cache/0902_sft_qwen35_2b_dat_ivcap-merged \
            "$CFS/vldat_experiments/0902_sft_qwen35_2b_dat_ivcap-merged") \
            || { echo "[ERROR] 找不到 2B DAT merged ckpt" >&2; exit 1; }
        # 2B 没训早退 arm, 只测 k=0
        DEFAULT_KS="0"
        ;;
    *)
        echo "[ERROR] SIZE 只支持 2B / 4B, 收到: $SIZE" >&2; exit 1 ;;
esac

EARLY_KS="${EARLY_KS:-$DEFAULT_KS}"
TASKS="${TASKS:-prefill memory batch_decode}"
# R 须为 96 的倍数 (patch16 x merge2 x hr_scale3)
RESOLUTIONS="${RESOLUTIONS:-672 1152 1344 2016 2688 3360}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
DECODE_LENS="${DECODE_LENS:-128}"
ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
STAMP="$(date +%m%d_%H%M)"

# ── 让平台的烧卡脚本安静下来 ────────────────────────────────────────────────
# auto_burn.py 用 GPU/CPU burn 防止 pod 被回收, 会把 util 顶到 70% / load 顶到 75,
# 测出来的延迟没有意义。它自己有 external-work 检测 (60s 一轮): 一旦看到别人的
# GPU compute 进程就停掉那张卡的 burn。所以这里只杀掉当前的 burn worker, 保留
# auto_burn 父进程 —— 测速期间它看到我们占卡就不会重启 burn, 测完 5 分钟後自动
# 恢复烧卡, pod 不会因为空闲被回收。
quiet_burn() {
    local pids
    pids=$(pgrep -f "auto_burn.py" 2>/dev/null || true)
    if [[ -z "$pids" ]]; then
        echo "[burn] 没有 auto_burn 进程, 跳过"
        return
    fi
    for p in $pids; do
        echo "[burn] 杀掉 auto_burn(pid=$p) 的 burn worker, 保留父进程"
        pkill -P "$p" 2>/dev/null || true
    done
    sleep 8
    echo "[burn] 当前 GPU 状态:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
               --format=csv,noheader | sed 's/^/  /'
    echo "[burn] load average: $(cut -d' ' -f1-3 /proc/loadavg)"
}

echo "════════════════════════════════════════════════════════════════"
echo "[cfg] SIZE=$SIZE  family=qwen3_5"
echo "[cfg] base=$BASE_MODEL"
echo "[cfg] dat =$DAT_CKPT"
echo "[cfg] tasks=$TASKS  early_ks=$EARLY_KS"
echo "[cfg] R=$RESOLUTIONS  B=$BATCH_SIZES  iters=$ITERS"
echo "[cfg] device=$CUDA_VISIBLE_DEVICES"
echo "════════════════════════════════════════════════════════════════"

quiet_burn

for K in $EARLY_KS; do
    TAG="q35_${SIZE}_k${K}_${STAMP}"
    echo
    echo "────────────────────────────────────────────────────────────────"
    echo "[run] hd_early_exit_k=$K  tag=$TAG"
    echo "────────────────────────────────────────────────────────────────"
    python test_inference_bench.py \
        --model-family qwen3_5 \
        --base-model "$BASE_MODEL" \
        --dat-ckpt "$DAT_CKPT" \
        --synthetic \
        --tasks $TASKS \
        --resolutions $RESOLUTIONS \
        --batch-sizes $BATCH_SIZES \
        --decode-lens $DECODE_LENS \
        --hd-early-k "$K" \
        --warmup "$WARMUP" --iters "$ITERS" \
        --tag "$TAG"
done

echo
echo "[done] 结果在 bench_outputs/bench_q35_${SIZE}_k*_${STAMP}.{json,md}"
