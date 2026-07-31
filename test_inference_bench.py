#!/usr/bin/env python3
"""
test_inference_bench.py

Qwen2.5-VL DAT 统一推理基准测试脚本（合并自 test_pareto_speed.py + test_dat_full_speed.py）。

═══════════════════════════════════════════════════════════════════════════════════
核心 Message
═══════════════════════════════════════════════════════════════════════════════════

DAT 通过把 LLM 输入序列从 Nq_hd 缩短到 Nq_lr（~3× 缩减），在工业实际推理场景下
带来速度和显存上的实质性提升。该优势不依赖底层 attention kernel 实现，在任何
推理引擎（transformers、vLLM、SGLang 等）下都成立。

═══════════════════════════════════════════════════════════════════════════════════
任务（--tasks）
═══════════════════════════════════════════════════════════════════════════════════

默认任务（核心，证明实际推理优势）：
  · prefill        — Base(R) vs DAT(HD=R) 的 E2E + 分解 (ViT_HD / ViT_LR / LLM)
  · batch_decode   — sweep (R, B, T) 测 decode throughput / TTFT / ITL
  · memory         — sweep (R, B) 测 peak memory / OOM 临界

可选任务（详细分析，需 --tasks 显式指定）：
  · layerwise      — 每个 decoder layer 的 CUDA event 计时（Base vs DAT）
  · vit_paths      — DAT-Sep / DAT-Fused / DAT-Shared / Fair-HD / LR-only 五路对比

═══════════════════════════════════════════════════════════════════════════════════
关键设计
═══════════════════════════════════════════════════════════════════════════════════

· 同一 DAT 模型，运行时切换 pixel_values_hd 决定走 DAT 路径还是 Base 路径，
  保证两路对比公平（同权重、同框架、同 attention backend）。

· 显存测量：每次 forward 前 reset_peak_memory_stats，forward 后取
  max_memory_allocated，反映真实推理峰值（含 KV cache、activation、临时 buffer）。

· OOM 容错：每个 (R, B) 包在 try/except 里，OOM 后 empty_cache 跳过，
  增量保存 JSON，防止中途 OOM 丢已测数据。

· 公平 batch：同一张图复制 B 次（控制变量），后续可换"多图模式"。

═══════════════════════════════════════════════════════════════════════════════════
用法
═══════════════════════════════════════════════════════════════════════════════════

# 默认核心 3 任务
python test_inference_bench.py --dat-ckpt /path/to/ckpt

# 只测 batch decode throughput
python test_inference_bench.py --tasks batch_decode \\
    --resolutions 1344 --batch-sizes 1 4 8 16 32

# 找 OOM 临界（极限分辨率 + batch）
python test_inference_bench.py --tasks memory \\
    --resolutions 1344 2016 2688 3360 --batch-sizes 1 4 16 32

# 长 decode latency（serving 场景）
python test_inference_bench.py --tasks batch_decode \\
    --decode-lens 32 128 512 2048 --batch-sizes 1

# 全部任务（含 layerwise 和 vit_paths）
python test_inference_bench.py --tasks all
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
import time
import json
import argparse
import contextlib
import traceback
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

import torch
from PIL import Image

# ─── 默认路径 (环境变量 MODEL_CACHE / VSTAR_DIR 可覆盖) ──────────────────────
MODEL_CACHE = os.environ.get('MODEL_CACHE', '/workspace/model_cache')
DEFAULT_BASE_MODEL = os.path.join(MODEL_CACHE, 'Qwen2.5-VL-3B-Instruct')
# 空 = 无训练 ckpt，直接从 base 权重构建 DAT (DAT 模块随机初始化，测速等价)
DEFAULT_DAT_CKPT   = ''
# vstar 目录不存在时自动回退到合成图 (等价于 --synthetic)
DEFAULT_VSTAR_DIR  = os.environ.get('VSTAR_DIR', 'data/vstar_bench')
DEFAULT_VSTAR_JSONL = os.path.join(DEFAULT_VSTAR_DIR, 'test_questions.jsonl')

# 从 base 构建时的 DAT 结构参数 (与 0701 实验一致; layers 按层数自动生成 1D-per-6)
DEFAULT_DAT_EXTRA_ARGS = {
    'grid_size': 20,
    'off_ksize': 3,
    'off_grps': 8,
    'inter_size': 128,
    'hr_scale': 3,
    'hd_proj': True,
    'use_intention_branch': True,
    'intention_as_gate': True,
    'use_spatial_attn_guide': False,
    'hd_gate_init': None,
    'hd_gate_freeze': False,
    'inject_lr_image': False,
    'image_hd_for_question': False,
    'use_fused_vit': False,
    'use_shared_vit': False,
    'hd_early_exit_k': 0,
}

# ─── 模型常量 ────────────────────────────────────────────────────────────────
PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR     = PATCH_SIZE * MERGE_SIZE   # 28 = pixels per merged token

DTYPE  = torch.bfloat16
DEVICE = 'cuda'

# ─── 默认 sweep 参数 ─────────────────────────────────────────────────────────
DEFAULT_RESOLUTIONS  = [672, 1008, 1344, 2016]
DEFAULT_BATCH_SIZES  = [1, 2, 4, 8, 16]
DEFAULT_DECODE_LENS  = [128]
DEFAULT_WARMUP       = 2
DEFAULT_ITERS        = 3
DEFAULT_LAYERWISE_R  = [1344]
DEFAULT_QUESTION     = "Describe this image in detail."

ALL_TASKS = ['prefill', 'batch_decode', 'memory', 'layerwise', 'vit_paths']
DEFAULT_TASKS = ['prefill', 'batch_decode', 'memory']


# ════════════════════════════════════════════════════════════════════════════════
# 通用工具
# ════════════════════════════════════════════════════════════════════════════════

def visual_tokens(inputs) -> int:
    """从 image_grid_thw 推算 visual token 数（merge 后）。"""
    thw = inputs["image_grid_thw"][0]
    t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
    return (h // MERGE_SIZE) * (w // MERGE_SIZE) * t


def make_text(processor, img: Image.Image, question: str = DEFAULT_QUESTION) -> str:
    """构造 chat-template 后的完整文本。"""
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": question},
    ]}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def benchmark_fn(fn, warmup: int = 2, iters: int = 3) -> float:
    """返回平均耗时 (ms/iter)，含 CUDA 同步。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def benchmark_with_memory(fn, warmup: int = 2, iters: int = 3) -> Tuple[float, float]:
    """
    返回 (mean_ms, peak_mem_mb)。每次 iter 前 reset_peak_memory_stats，
    取所有 iter 的 max_memory_allocated 的最大值作为峰值。
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    peaks = []
    times = []
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        peaks.append(torch.cuda.max_memory_allocated() / (1024 ** 2))   # MB

    return sum(times) / len(times), max(peaks)


@contextlib.contextmanager
def oom_guard(label: str = ""):
    """
    OOM 容错上下文。捕获 OOM 后清显存、yield None。
    用法:
      with oom_guard("R=2016 B=32") as guard:
          ... do work ...
          guard['ok'] = True
          guard['result'] = {...}
      if not guard['ok']: 跳过
    """
    state = {'ok': False, 'oom': False, 'result': None, 'error': None}
    try:
        yield state
    except torch.cuda.OutOfMemoryError as e:
        state['oom'] = True
        state['error'] = f"OOM: {str(e)[:200]}"
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if label:
            print(f"  [OOM] {label}")
    except Exception as e:
        state['error'] = f"{type(e).__name__}: {str(e)[:200]}"
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if label:
            print(f"  [ERR] {label}: {state['error']}")
            traceback.print_exc(limit=3)


def gpu_info() -> Dict[str, Any]:
    """收集 GPU 与环境元信息。"""
    info = {
        'pytorch':  torch.__version__,
        'cuda':     torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_total_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
    }
    return info


def _set_vit_path(dat_model, mode: str):
    """运行时切换 DAT ViT 路径。forward() 每次都从 config 读 flag，立即生效。

    mode: 'separate' | 'fused' | 'shared'
    """
    cfg = dat_model.config.dat_extra_args
    cfg['use_fused_vit']  = (mode == 'fused')
    cfg['use_shared_vit'] = (mode == 'shared')


# ════════════════════════════════════════════════════════════════════════════════
# 图像处理
# ════════════════════════════════════════════════════════════════════════════════

def process_base(processor, img, text, R, batch=1, device=DEVICE):
    """
    Base(R): min=max=R² 像素，含完整问题文本。
    若 batch > 1，将同一图与文本复制 batch 份。
    """
    images = [img] * batch
    texts  = [text] * batch
    inp = processor(images=images, text=texts, return_tensors="pt",
                    padding=True, min_pixels=R * R, max_pixels=R * R)
    return {k: v.to(device) for k, v in inp.items()}


def process_dat(processor, img, text, R, hr_scale, batch=1, device=DEVICE):
    """
    DAT 双路:
      inp_lr  : LR 分辨率 (R/hr_scale)²，含完整问题文本
      inp_hd  : HD 分辨率 R²，仅取 pixel_values/image_grid_thw（无文本）
    若 batch > 1，将同一图复制 batch 份。
    """
    lr = R // hr_scale
    images = [img] * batch
    texts  = [text] * batch
    inp_lr = processor(images=images, text=texts, return_tensors="pt",
                       padding=True, min_pixels=lr * lr, max_pixels=lr * lr)

    # HD: 仅图像部分，文本占位即可（不进 LLM 路径）
    inp_hd = processor(images=images, text=["<|im_start|>"] * batch, return_tensors="pt",
                       padding=True, min_pixels=R * R, max_pixels=R * R)
    return (
        {k: v.to(device) for k, v in inp_lr.items()},
        {k: v.to(device) for k, v in inp_hd.items()},
    )


def process_fair_hd(processor, img, text, R, batch=1, device=DEVICE):
    """
    Fair-HD baseline（仅 vit_paths 任务用）:
    HD 分辨率 R²，但含完整问题文本（与 DAT inp_lr 同问题）。
    生成 Nq_hd >> Nq_lr 的较长输入序列，作为 naive HD baseline。
    """
    images = [img] * batch
    texts  = [text] * batch
    inp = processor(images=images, text=texts, return_tensors="pt",
                    padding=True, min_pixels=R * R, max_pixels=R * R)
    return {k: v.to(device) for k, v in inp.items()}


def make_synthetic_image(R: int) -> Image.Image:
    """合成一张 R×R 图（控制分辨率精确）。"""
    return Image.new("RGB", (R, R), color=(100, 149, 237))


# ════════════════════════════════════════════════════════════════════════════════
# Task 1: Prefill (E2E + Breakdown)
#   合并自 test_pareto_speed.py 的 ① E2E Prefill + ② Breakdown
# ════════════════════════════════════════════════════════════════════════════════

def task_prefill(dat_model, processor, img, R, hr_scale, warmup, iters,
                 qwen_model=None):
    """单分辨率 R 上的 Prefill 测量。

    qwen_model: 原生 Qwen2_5_VLForConditionalGeneration (官方类)。提供时额外测
    t_qwen_e2e_ms，用于对照 DAT 类的 base 路径是否存在类本身的额外开销。

    返回:
      {
        'R', 'LR', 'HD_tokens', 'LR_tokens', 'nq_base', 'nq_lr',
        't_base_e2e_ms', 't_dat_e2e_ms', 't_dat_e2e_fused_ms',
        't_qwen_e2e_ms', 'speedup', 'speedup_fused', 'speedup_fused_vs_qwen',
        't_vit_hd_ms', 't_vit_lr_ms', 't_llm_dat_pre_ms',
        'llm_base_est_ms', 'llm_dat_est_ms',
        'mem_base_mb', 'mem_dat_mb',
      }
    """
    text = make_text(processor, img)
    inp_base       = process_base(processor, img, text, R, batch=1)
    inp_lr, inp_hd = process_dat(processor, img, text, R, hr_scale, batch=1)

    pv_base  = inp_base['pixel_values'].to(DTYPE)
    thw_base = inp_base['image_grid_thw']
    iids_b   = inp_base['input_ids']
    amask_b  = inp_base.get('attention_mask')

    pv_lr   = inp_lr['pixel_values'].to(DTYPE)
    thw_lr  = inp_lr['image_grid_thw']
    iids_l  = inp_lr['input_ids']
    amask_l = inp_lr.get('attention_mask')

    pv_hd   = inp_hd['pixel_values'].to(DTYPE)
    thw_hd  = inp_hd['image_grid_thw']

    nq_base = iids_b.shape[1]
    nq_lr   = iids_l.shape[1]
    n_vis_b = visual_tokens(inp_base)
    n_vis_l = visual_tokens(inp_lr)
    n_vis_h = visual_tokens(inp_hd)

    # 预计算 HD features，用于 LLM_DAT-only 测量（排除 HD ViT）
    with torch.no_grad():
        hd_feats = dat_model._generate_hd_features(pv_hd, thw_hd)

    def fwd_base_e2e():
        with torch.no_grad():
            dat_model(input_ids=iids_b, attention_mask=amask_b,
                      pixel_values=pv_base, image_grid_thw=thw_base,
                      pixel_values_hd=None, image_grid_thw_hd=None,
                      use_cache=False)

    def fwd_dat_e2e():
        with torch.no_grad():
            dat_model(input_ids=iids_l, attention_mask=amask_l,
                      pixel_values=pv_lr, image_grid_thw=thw_lr,
                      pixel_values_hd=pv_hd, image_grid_thw_hd=thw_hd,
                      use_cache=False)

    def fwd_vit_hd():
        with torch.no_grad():
            dat_model._generate_hd_features(pv_hd, thw_hd)

    def fwd_vit_lr():
        with torch.no_grad():
            dat_model._generate_hd_features(pv_lr, thw_lr)

    def fwd_llm_dat_pre():
        # LR ViT + LLM_DAT（HD 已预算）
        with torch.no_grad():
            dat_model(input_ids=iids_l, attention_mask=amask_l,
                      pixel_values=pv_lr, image_grid_thw=thw_lr,
                      image_hd_features=hd_feats, use_cache=False)

    def fwd_qwen_e2e():
        # 原生 Qwen2.5-VL 官方类, 同 Base(R) 输入
        with torch.no_grad():
            qwen_model(input_ids=iids_b, attention_mask=amask_b,
                       pixel_values=pv_base, image_grid_thw=thw_base,
                       use_cache=False)

    # 联合 warmup（所有路径）
    for _ in range(warmup):
        fwd_base_e2e(); fwd_dat_e2e(); fwd_vit_hd(); fwd_vit_lr(); fwd_llm_dat_pre()
        if qwen_model is not None:
            fwd_qwen_e2e()
    torch.cuda.synchronize()

    # 计时 + 显存（E2E 两路单独测显存，分解项只测时间）
    t_base_e2e, mem_base = benchmark_with_memory(fwd_base_e2e,    warmup=0, iters=iters)
    t_dat_e2e,  mem_dat  = benchmark_with_memory(fwd_dat_e2e,     warmup=0, iters=iters)

    # DAT E2E with fused ViT (LR+HD 单次 visual() 调用)
    _set_vit_path(dat_model, 'fused')
    t_dat_e2e_fused, mem_dat_fused = benchmark_with_memory(
        fwd_dat_e2e, warmup=max(1, warmup), iters=iters)
    _set_vit_path(dat_model, 'separate')

    t_vit_hd      = benchmark_fn(fwd_vit_hd,      warmup=0, iters=iters)
    t_vit_lr      = benchmark_fn(fwd_vit_lr,      warmup=0, iters=iters)
    t_llm_dat_pre = benchmark_fn(fwd_llm_dat_pre, warmup=0, iters=iters)

    t_qwen_e2e, mem_qwen = (
        benchmark_with_memory(fwd_qwen_e2e, warmup=0, iters=iters)
        if qwen_model is not None else (None, None)
    )

    llm_base = max(0.0, t_base_e2e    - t_vit_hd)
    llm_dat  = max(0.0, t_llm_dat_pre - t_vit_lr)

    return {
        'R': R, 'LR': R // hr_scale,
        'HD_tokens': n_vis_h, 'LR_tokens': n_vis_l, 'Base_vis_tokens': n_vis_b,
        'nq_base': nq_base, 'nq_lr': nq_lr,
        't_base_e2e_ms':    t_base_e2e,
        't_dat_e2e_ms':     t_dat_e2e,
        't_dat_e2e_fused_ms': t_dat_e2e_fused,
        't_qwen_e2e_ms':    t_qwen_e2e,
        'mem_qwen_mb':      mem_qwen,
        'speedup':          t_base_e2e / t_dat_e2e if t_dat_e2e > 0 else 0,
        'speedup_fused':    t_base_e2e / t_dat_e2e_fused if t_dat_e2e_fused > 0 else 0,
        'speedup_fused_vs_qwen': (
            t_qwen_e2e / t_dat_e2e_fused
            if t_qwen_e2e and t_dat_e2e_fused > 0 else None),
        't_vit_hd_ms':      t_vit_hd,
        't_vit_lr_ms':      t_vit_lr,
        't_llm_dat_pre_ms': t_llm_dat_pre,
        'llm_base_est_ms':  llm_base,
        'llm_dat_est_ms':   llm_dat,
        'llm_speedup':      llm_base / llm_dat if llm_dat > 0 else 0,
        'mem_base_mb':      mem_base,
        'mem_dat_mb':       mem_dat,
        'mem_dat_fused_mb': mem_dat_fused,
    }


def run_prefill_sweep(dat_model, processor, img, resolutions, hr_scale,
                      warmup, iters, save_cb=None, qwen_model=None):
    """对所有分辨率跑 prefill task，返回 list of result dicts。"""
    results = []
    for R in resolutions:
        unit = FACTOR * hr_scale
        if R % unit != 0:
            print(f"  [skip] R={R} 非 {unit} 的倍数")
            continue

        with oom_guard(f"prefill R={R}") as guard:
            res = task_prefill(dat_model, processor, img, R, hr_scale,
                               warmup=warmup, iters=iters, qwen_model=qwen_model)
            guard['ok'] = True
            guard['result'] = res

        entry = {'R': R, 'task': 'prefill'}
        if guard['ok']:
            entry.update(guard['result'])
            r = guard['result']
            qwen_str = (f"  Qwen={r['t_qwen_e2e_ms']:>7.1f}ms"
                        f" (fused vs Qwen: {r['speedup_fused_vs_qwen']:.2f}x)"
                        if r.get('t_qwen_e2e_ms') else "")
            print(f"  R={R:>5}  Nq_base={r['nq_base']:>5}  Nq_lr={r['nq_lr']:>5}"
                  f"  Base={r['t_base_e2e_ms']:>7.1f}ms  DAT={r['t_dat_e2e_ms']:>7.1f}ms"
                  f" ({r['speedup']:.2f}x)"
                  f"  DAT-fused={r['t_dat_e2e_fused_ms']:>7.1f}ms ({r['speedup_fused']:.2f}x)"
                  f"{qwen_str}"
                  f"  Mem Base={r['mem_base_mb']/1024:>5.2f}GB DAT={r['mem_dat_mb']/1024:>5.2f}GB")
        else:
            entry['oom'] = guard['oom']
            entry['error'] = guard['error']

        results.append(entry)
        if save_cb is not None:
            save_cb()
    return results


# ════════════════════════════════════════════════════════════════════════════════
# Task 2: Batch Decode (NEW)
#   sweep (R, B, T) → tokens/s, samples/s, TTFT (prefill), ITL (per-token decode)
# ════════════════════════════════════════════════════════════════════════════════

def _measure_generate(model, inputs, gen_kwargs, warmup=1, iters=2):
    """
    测 generate 全流程：返回 (mean_total_ms, peak_mem_mb)。
    inputs / gen_kwargs 须已拷贝到 GPU。
    """
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    times, peaks = [], []
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        peaks.append(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return sum(times) / len(times), max(peaks)


def _measure_prefill_only(model, fwd_fn, warmup=1, iters=2):
    """测单次 forward (prefill) 时间。"""
    for _ in range(warmup):
        fwd_fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd_fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def task_batch_decode(dat_model, processor, img, R, B, T, hr_scale,
                      warmup=1, iters=2, qwen_model=None):
    """
    单个 (R, B, T) 配置上的 decode throughput 测量。
    DAT 与 Base 各跑一次（可选原生 Qwen 一次），返回 dict 列表。
    """
    text = make_text(processor, img)
    eos_id = processor.tokenizer.eos_token_id

    # 用 do_sample=False + min/max_new_tokens=T 强制生成固定 T 个 token，
    # 屏蔽 eos 提前终止，确保数字可比。
    gen_kwargs = dict(
        max_new_tokens=T, min_new_tokens=T,
        do_sample=False, temperature=1.0, top_p=1.0,
        use_cache=True,
        pad_token_id=eos_id,
    )

    out_records = []

    # ── Qwen 原生（官方类, 同 Base 输入）────────────────────────────────────
    if qwen_model is not None:
        with oom_guard(f"batch_decode Qwen R={R} B={B} T={T}") as guard:
            inp = process_base(processor, img, text, R, batch=B)
            nq  = inp['input_ids'].shape[1]

            def prefill_only():
                with torch.no_grad():
                    qwen_model(input_ids=inp['input_ids'],
                               attention_mask=inp.get('attention_mask'),
                               pixel_values=inp['pixel_values'].to(DTYPE),
                               image_grid_thw=inp['image_grid_thw'],
                               use_cache=True)

            gen_inputs = dict(
                input_ids=inp['input_ids'],
                attention_mask=inp.get('attention_mask'),
                pixel_values=inp['pixel_values'].to(DTYPE),
                image_grid_thw=inp['image_grid_thw'],
            )

            t_prefill = _measure_prefill_only(qwen_model, prefill_only,
                                              warmup=warmup, iters=iters)
            t_total, mem_peak = _measure_generate(qwen_model, gen_inputs,
                                                  gen_kwargs,
                                                  warmup=warmup, iters=iters)
            t_decode = max(0.0, t_total - t_prefill)

            guard['ok'] = True
            guard['result'] = dict(
                R=R, B=B, T=T, model='qwen', nq=nq, decode_tokens=B * T,
                ttft_ms=t_prefill,
                decode_ms=t_decode,
                total_ms=t_total,
                itl_ms_per_token=t_decode / (B * T) if B * T > 0 else 0,
                tokens_per_sec=(B * T) / (t_decode / 1000) if t_decode > 0 else 0,
                samples_per_sec=B / (t_total / 1000) if t_total > 0 else 0,
                peak_mem_mb=mem_peak,
            )

        if guard['ok']:
            r = guard['result']
            print(f"  Qwen  R={R:>5} B={B:>3} T={T:>4} Nq={r['nq']:>5}"
                  f"  TTFT={r['ttft_ms']:>7.1f}ms  decode={r['decode_ms']:>7.1f}ms"
                  f"  ITL={r['itl_ms_per_token']:>5.1f}ms/tok"
                  f"  tok/s={r['tokens_per_sec']:>7.1f}"
                  f"  mem={r['peak_mem_mb']/1024:>5.2f}GB")
            out_records.append(r)
        else:
            out_records.append(dict(R=R, B=B, T=T, model='qwen',
                                    oom=guard['oom'], error=guard['error']))

    # ── Base: pixel_values_hd=None（DAT 模型退化为 Qwen2.5-VL Base）─────────────
    with oom_guard(f"batch_decode Base R={R} B={B} T={T}") as guard:
        inp = process_base(processor, img, text, R, batch=B)
        nq  = inp['input_ids'].shape[1]

        def prefill_only():
            with torch.no_grad():
                dat_model(input_ids=inp['input_ids'],
                          attention_mask=inp.get('attention_mask'),
                          pixel_values=inp['pixel_values'].to(DTYPE),
                          image_grid_thw=inp['image_grid_thw'],
                          pixel_values_hd=None, image_grid_thw_hd=None,
                          use_cache=True)

        gen_inputs = dict(
            input_ids=inp['input_ids'],
            attention_mask=inp.get('attention_mask'),
            pixel_values=inp['pixel_values'].to(DTYPE),
            image_grid_thw=inp['image_grid_thw'],
            pixel_values_hd=None, image_grid_thw_hd=None,
        )

        t_prefill = _measure_prefill_only(dat_model, prefill_only,
                                          warmup=warmup, iters=iters)
        t_total, mem_peak = _measure_generate(dat_model, gen_inputs, gen_kwargs,
                                              warmup=warmup, iters=iters)
        t_decode = max(0.0, t_total - t_prefill)

        guard['ok'] = True
        guard['result'] = dict(
            R=R, B=B, T=T, model='base', nq=nq, decode_tokens=B * T,
            ttft_ms=t_prefill,
            decode_ms=t_decode,
            total_ms=t_total,
            itl_ms_per_token=t_decode / (B * T) if B * T > 0 else 0,
            tokens_per_sec=(B * T) / (t_decode / 1000) if t_decode > 0 else 0,
            samples_per_sec=B / (t_total / 1000) if t_total > 0 else 0,
            peak_mem_mb=mem_peak,
        )

    if guard['ok']:
        r = guard['result']
        print(f"  Base  R={R:>5} B={B:>3} T={T:>4} Nq={r['nq']:>5}"
              f"  TTFT={r['ttft_ms']:>7.1f}ms  decode={r['decode_ms']:>7.1f}ms"
              f"  ITL={r['itl_ms_per_token']:>5.1f}ms/tok"
              f"  tok/s={r['tokens_per_sec']:>7.1f}"
              f"  mem={r['peak_mem_mb']/1024:>5.2f}GB")
        out_records.append(r)
    else:
        out_records.append(dict(R=R, B=B, T=T, model='base',
                                oom=guard['oom'], error=guard['error']))

    # ── DAT: pixel_values_hd 启用 ────────────────────────────────────────────
    with oom_guard(f"batch_decode DAT R={R} B={B} T={T}") as guard:
        inp_lr, inp_hd = process_dat(processor, img, text, R, hr_scale, batch=B)
        nq = inp_lr['input_ids'].shape[1]

        def prefill_only():
            with torch.no_grad():
                dat_model(input_ids=inp_lr['input_ids'],
                          attention_mask=inp_lr.get('attention_mask'),
                          pixel_values=inp_lr['pixel_values'].to(DTYPE),
                          image_grid_thw=inp_lr['image_grid_thw'],
                          pixel_values_hd=inp_hd['pixel_values'].to(DTYPE),
                          image_grid_thw_hd=inp_hd['image_grid_thw'],
                          use_cache=True)

        gen_inputs = dict(
            input_ids=inp_lr['input_ids'],
            attention_mask=inp_lr.get('attention_mask'),
            pixel_values=inp_lr['pixel_values'].to(DTYPE),
            image_grid_thw=inp_lr['image_grid_thw'],
            pixel_values_hd=inp_hd['pixel_values'].to(DTYPE),
            image_grid_thw_hd=inp_hd['image_grid_thw'],
        )

        t_prefill = _measure_prefill_only(dat_model, prefill_only,
                                          warmup=warmup, iters=iters)
        t_total, mem_peak = _measure_generate(dat_model, gen_inputs, gen_kwargs,
                                              warmup=warmup, iters=iters)
        t_decode = max(0.0, t_total - t_prefill)

        guard['ok'] = True
        guard['result'] = dict(
            R=R, B=B, T=T, model='dat', nq=nq, decode_tokens=B * T,
            ttft_ms=t_prefill,
            decode_ms=t_decode,
            total_ms=t_total,
            itl_ms_per_token=t_decode / (B * T) if B * T > 0 else 0,
            tokens_per_sec=(B * T) / (t_decode / 1000) if t_decode > 0 else 0,
            samples_per_sec=B / (t_total / 1000) if t_total > 0 else 0,
            peak_mem_mb=mem_peak,
        )

    if guard['ok']:
        r = guard['result']
        print(f"  DAT   R={R:>5} B={B:>3} T={T:>4} Nq={r['nq']:>5}"
              f"  TTFT={r['ttft_ms']:>7.1f}ms  decode={r['decode_ms']:>7.1f}ms"
              f"  ITL={r['itl_ms_per_token']:>5.1f}ms/tok"
              f"  tok/s={r['tokens_per_sec']:>7.1f}"
              f"  mem={r['peak_mem_mb']/1024:>5.2f}GB")
        out_records.append(r)
    else:
        out_records.append(dict(R=R, B=B, T=T, model='dat',
                                oom=guard['oom'], error=guard['error']))

    return out_records


def run_batch_decode_sweep(dat_model, processor, img, resolutions, batch_sizes,
                           decode_lens, hr_scale, warmup, iters, save_cb=None,
                           qwen_model=None):
    """sweep (R, B, T)"""
    results = []
    for R in resolutions:
        unit = FACTOR * hr_scale
        if R % unit != 0:
            print(f"  [skip] R={R} 非 {unit} 的倍数")
            continue
        for B in batch_sizes:
            for T in decode_lens:
                print(f"\n  ── (R={R}, B={B}, T={T}) ──")
                recs = task_batch_decode(dat_model, processor, img, R, B, T,
                                         hr_scale, warmup=warmup, iters=iters,
                                         qwen_model=qwen_model)
                results.extend(recs)
                if save_cb is not None:
                    save_cb()
    return results


# ════════════════════════════════════════════════════════════════════════════════
# Task 3: Memory & OOM 临界 (NEW)
#   sweep (R, B) 测 forward-only 峰值显存与 OOM 临界
#   与 batch_decode 不同，仅测单次 forward（不 generate），更接近"模型本身的开销"
# ════════════════════════════════════════════════════════════════════════════════

def task_memory(dat_model, processor, img, R, B, hr_scale, warmup=1, iters=2):
    """单 (R, B) 上 forward-only 显存测量。"""
    text = make_text(processor, img)
    out_records = []

    # ── Base ────────────────────────────────────────────────────────────────
    with oom_guard(f"memory Base R={R} B={B}") as guard:
        inp = process_base(processor, img, text, R, batch=B)
        nq  = inp['input_ids'].shape[1]
        n_vis = visual_tokens(inp)

        def fwd():
            with torch.no_grad():
                dat_model(input_ids=inp['input_ids'],
                          attention_mask=inp.get('attention_mask'),
                          pixel_values=inp['pixel_values'].to(DTYPE),
                          image_grid_thw=inp['image_grid_thw'],
                          pixel_values_hd=None, image_grid_thw_hd=None,
                          use_cache=False)

        t_ms, mem_mb = benchmark_with_memory(fwd, warmup=warmup, iters=iters)
        guard['ok'] = True
        guard['result'] = dict(R=R, B=B, model='base', nq=nq, vis_tokens=n_vis,
                               t_ms=t_ms, peak_mem_mb=mem_mb)

    if guard['ok']:
        r = guard['result']
        print(f"  Base  R={R:>5} B={B:>3} Nq={r['nq']:>5}"
              f"  t={r['t_ms']:>7.1f}ms  mem={r['peak_mem_mb']/1024:>5.2f}GB")
        out_records.append(r)
    else:
        out_records.append(dict(R=R, B=B, model='base',
                                oom=guard['oom'], error=guard['error']))

    # ── DAT ─────────────────────────────────────────────────────────────────
    with oom_guard(f"memory DAT R={R} B={B}") as guard:
        inp_lr, inp_hd = process_dat(processor, img, text, R, hr_scale, batch=B)
        nq = inp_lr['input_ids'].shape[1]
        n_vis_lr = visual_tokens(inp_lr)

        def fwd():
            with torch.no_grad():
                dat_model(input_ids=inp_lr['input_ids'],
                          attention_mask=inp_lr.get('attention_mask'),
                          pixel_values=inp_lr['pixel_values'].to(DTYPE),
                          image_grid_thw=inp_lr['image_grid_thw'],
                          pixel_values_hd=inp_hd['pixel_values'].to(DTYPE),
                          image_grid_thw_hd=inp_hd['image_grid_thw'],
                          use_cache=False)

        t_ms, mem_mb = benchmark_with_memory(fwd, warmup=warmup, iters=iters)
        guard['ok'] = True
        guard['result'] = dict(R=R, B=B, model='dat', nq=nq, vis_tokens=n_vis_lr,
                               t_ms=t_ms, peak_mem_mb=mem_mb)

    if guard['ok']:
        r = guard['result']
        print(f"  DAT   R={R:>5} B={B:>3} Nq={r['nq']:>5}"
              f"  t={r['t_ms']:>7.1f}ms  mem={r['peak_mem_mb']/1024:>5.2f}GB")
        out_records.append(r)
    else:
        out_records.append(dict(R=R, B=B, model='dat',
                                oom=guard['oom'], error=guard['error']))

    return out_records


def run_memory_sweep(dat_model, processor, img, resolutions, batch_sizes,
                     hr_scale, warmup, iters, save_cb=None):
    """sweep (R, B) for forward-only memory."""
    results = []
    for R in resolutions:
        unit = FACTOR * hr_scale
        if R % unit != 0:
            print(f"  [skip] R={R} 非 {unit} 的倍数")
            continue
        for B in batch_sizes:
            print(f"\n  ── (R={R}, B={B}) ──")
            recs = task_memory(dat_model, processor, img, R, B, hr_scale,
                               warmup=warmup, iters=iters)
            results.extend(recs)
            if save_cb is not None:
                save_cb()
    return results


# ════════════════════════════════════════════════════════════════════════════════
# Task 4: Layerwise (optional)
#   合并自 test_pareto_speed.py ③ + test_dat_full_speed.py C
# ════════════════════════════════════════════════════════════════════════════════

def layerwise_timing(model, fwd_fn, n_iters=5):
    """CUDA event hook 逐层计时。"""
    layers = model.model.language_model.layers
    N = len(layers)
    all_times = defaultdict(list)
    for _ in range(n_iters):
        evts_s = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
        evts_e = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
        handles = []
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(
                lambda m, a, _i=i: evts_s[_i].record()))
            handles.append(layer.register_forward_hook(
                lambda m, a, o, _i=i: evts_e[_i].record()))
        fwd_fn()
        torch.cuda.synchronize()
        for i in range(N):
            all_times[i].append(evts_s[i].elapsed_time(evts_e[i]))
        for h in handles:
            h.remove()
    return {i: sum(ts) / len(ts) for i, ts in all_times.items()}


def task_layerwise(dat_model, processor, img, R, hr_scale, dat_layer_indices,
                   warmup=2, n_iters=5):
    """对分辨率 R 做 Base vs DAT 的逐层计时。"""
    text = make_text(processor, img)
    inp_base       = process_base(processor, img, text, R, batch=1)
    inp_lr, inp_hd = process_dat(processor, img, text, R, hr_scale, batch=1)

    pv_base  = inp_base['pixel_values'].to(DTYPE)
    thw_base = inp_base['image_grid_thw']
    iids_b   = inp_base['input_ids']
    amask_b  = inp_base.get('attention_mask')

    pv_lr   = inp_lr['pixel_values'].to(DTYPE)
    thw_lr  = inp_lr['image_grid_thw']
    iids_l  = inp_lr['input_ids']
    amask_l = inp_lr.get('attention_mask')

    pv_hd   = inp_hd['pixel_values'].to(DTYPE)
    thw_hd  = inp_hd['image_grid_thw']

    with torch.no_grad():
        hd_feats = dat_model._generate_hd_features(pv_hd, thw_hd)

    def fwd_base():
        with torch.no_grad():
            dat_model(input_ids=iids_b, attention_mask=amask_b,
                      pixel_values=pv_base, image_grid_thw=thw_base,
                      pixel_values_hd=None, image_grid_thw_hd=None,
                      use_cache=False)

    def fwd_dat():
        with torch.no_grad():
            dat_model(input_ids=iids_l, attention_mask=amask_l,
                      pixel_values=pv_lr, image_grid_thw=thw_lr,
                      image_hd_features=hd_feats, use_cache=False)

    for _ in range(warmup):
        fwd_base(); fwd_dat()
    torch.cuda.synchronize()

    t_base_lw = layerwise_timing(dat_model, fwd_base, n_iters=n_iters)
    t_dat_lw  = layerwise_timing(dat_model, fwd_dat,  n_iters=n_iters)

    N        = len(dat_model.model.language_model.layers)
    dat_set  = set(dat_layer_indices)
    total_b  = sum(t_base_lw.values())
    total_d  = sum(t_dat_lw.values())

    layer_rows = []
    for i in range(N):
        layer_rows.append({
            'layer': i,
            'type': 'DAT' if i in dat_set else 'Std',
            't_base_ms': t_base_lw.get(i, 0.0),
            't_dat_ms':  t_dat_lw.get(i, 0.0),
            'diff_ms':   t_dat_lw.get(i, 0.0) - t_base_lw.get(i, 0.0),
        })

    return {
        'R': R, 'nq_base': iids_b.shape[1], 'nq_lr': iids_l.shape[1],
        'total_base_ms': total_b, 'total_dat_ms': total_d,
        'dat_layer_indices': sorted(list(dat_set)),
        'layers': layer_rows,
    }


def run_layerwise_sweep(dat_model, processor, img, resolutions, hr_scale,
                        dat_layer_indices, warmup, n_iters, save_cb=None):
    results = []
    for R in resolutions:
        unit = FACTOR * hr_scale
        if R % unit != 0:
            print(f"  [skip] R={R} 非 {unit} 的倍数")
            continue
        with oom_guard(f"layerwise R={R}") as guard:
            res = task_layerwise(dat_model, processor, img, R, hr_scale,
                                 dat_layer_indices, warmup=warmup, n_iters=n_iters)
            guard['ok'] = True
            guard['result'] = res

        entry = {'R': R, 'task': 'layerwise'}
        if guard['ok']:
            entry.update(guard['result'])
            r = guard['result']
            print(f"  R={R:>5}  Base total={r['total_base_ms']:>7.1f}ms"
                  f"  DAT total={r['total_dat_ms']:>7.1f}ms"
                  f"  diff={r['total_dat_ms']-r['total_base_ms']:+.1f}ms")
        else:
            entry['oom'] = guard['oom']
            entry['error'] = guard['error']
        results.append(entry)
        if save_cb is not None:
            save_cb()
    return results


# ════════════════════════════════════════════════════════════════════════════════
# Task 5: ViT Paths (optional)
#   来自 test_dat_full_speed.py Part B
# ════════════════════════════════════════════════════════════════════════════════

def task_vit_paths(dat_model, processor, img, R, hr_scale, warmup=2, iters=3):
    """五路对比：DAT-Sep / DAT-Fused / DAT-Shared / Fair-HD / LR-only。"""
    text = make_text(processor, img)
    inp_lr, inp_hd = process_dat(processor, img, text, R, hr_scale, batch=1)
    inp_hd_full    = process_fair_hd(processor, img, text, R, batch=1)

    pv_lr  = inp_lr['pixel_values'].to(DTYPE);  thw_lr  = inp_lr['image_grid_thw']
    pv_hd  = inp_hd['pixel_values'].to(DTYPE);  thw_hd  = inp_hd['image_grid_thw']
    pv_hdf = inp_hd_full['pixel_values'].to(DTYPE); thw_hdf = inp_hd_full['image_grid_thw']

    iids_l    = inp_lr['input_ids'];        amask_l    = inp_lr.get('attention_mask')
    iids_hdf  = inp_hd_full['input_ids'];   amask_hdf  = inp_hd_full.get('attention_mask')

    nq_lr  = iids_l.shape[1]
    nq_hd  = iids_hdf.shape[1]

    def fwd_hd_enc():
        with torch.no_grad():
            dat_model._generate_hd_features(pv_hd, thw_hd)

    def make_fwd_dat(mode):
        def _f():
            _set_vit_path(dat_model, mode)
            with torch.no_grad():
                dat_model(input_ids=iids_l, attention_mask=amask_l,
                          pixel_values=pv_lr, image_grid_thw=thw_lr,
                          pixel_values_hd=pv_hd, image_grid_thw_hd=thw_hd,
                          use_cache=False)
        return _f

    def fwd_fair_hd():
        _set_vit_path(dat_model, 'separate')
        with torch.no_grad():
            dat_model(input_ids=iids_hdf, attention_mask=amask_hdf,
                      pixel_values=pv_hdf, image_grid_thw=thw_hdf,
                      pixel_values_hd=None, image_grid_thw_hd=None,
                      use_cache=False)

    def fwd_lr_only():
        _set_vit_path(dat_model, 'separate')
        with torch.no_grad():
            dat_model(input_ids=iids_l, attention_mask=amask_l,
                      pixel_values=pv_lr, image_grid_thw=thw_lr,
                      pixel_values_hd=None, image_grid_thw_hd=None,
                      use_cache=False)

    t_hd_enc = benchmark_fn(fwd_hd_enc,               warmup=warmup, iters=iters)
    t_sep    = benchmark_fn(make_fwd_dat('separate'), warmup=warmup, iters=iters)
    t_fused  = benchmark_fn(make_fwd_dat('fused'),    warmup=warmup, iters=iters)
    t_shared = benchmark_fn(make_fwd_dat('shared'),   warmup=warmup, iters=iters)
    t_fair   = benchmark_fn(fwd_fair_hd,              warmup=warmup, iters=iters)
    t_lr     = benchmark_fn(fwd_lr_only,              warmup=warmup, iters=iters)

    _set_vit_path(dat_model, 'separate')   # 恢复

    return {
        'R': R, 'nq_lr': nq_lr, 'nq_hd': nq_hd,
        't_hd_enc_ms':     t_hd_enc,
        't_dat_sep_ms':    t_sep,
        't_dat_fused_ms':  t_fused,
        't_dat_shared_ms': t_shared,
        't_fair_hd_ms':    t_fair,
        't_lr_only_ms':    t_lr,
        'shared_over_sep': t_shared / t_sep if t_sep > 0 else 0,
        'shared_over_hd_enc': t_shared / t_hd_enc if t_hd_enc > 0 else 0,
    }


def run_vit_paths_sweep(dat_model, processor, img, resolutions, hr_scale,
                        warmup, iters, save_cb=None):
    results = []
    for R in resolutions:
        unit = FACTOR * hr_scale
        if R % unit != 0:
            print(f"  [skip] R={R} 非 {unit} 的倍数")
            continue
        with oom_guard(f"vit_paths R={R}") as guard:
            res = task_vit_paths(dat_model, processor, img, R, hr_scale,
                                 warmup=warmup, iters=iters)
            guard['ok'] = True
            guard['result'] = res

        entry = {'R': R, 'task': 'vit_paths'}
        if guard['ok']:
            entry.update(guard['result'])
            r = guard['result']
            print(f"  R={R:>5}"
                  f"  Sep={r['t_dat_sep_ms']:>7.1f}"
                  f"  Fused={r['t_dat_fused_ms']:>7.1f}"
                  f"  Shared={r['t_dat_shared_ms']:>7.1f}"
                  f"  Fair-HD={r['t_fair_hd_ms']:>7.1f}"
                  f"  LR-only={r['t_lr_only_ms']:>7.1f}  (ms)")
        else:
            entry['oom'] = guard['oom']
            entry['error'] = guard['error']
        results.append(entry)
        if save_cb is not None:
            save_cb()
    return results


# ════════════════════════════════════════════════════════════════════════════════
# 输出: JSON + Markdown
# ════════════════════════════════════════════════════════════════════════════════

def write_json(path: str, payload: Dict[str, Any]):
    """原子写：写到 .tmp 后 rename。"""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _md_table(rows: List[List[str]], headers: List[str]) -> str:
    """生成 Markdown 表格字符串。"""
    if not rows:
        return "_(empty)_\n"
    out = ['| ' + ' | '.join(headers) + ' |']
    out.append('| ' + ' | '.join('---' for _ in headers) + ' |')
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out) + '\n'


def render_markdown(payload: Dict[str, Any]) -> str:
    """从 payload 生成 Markdown 报告。"""
    md = []
    meta = payload.get('metadata', {})
    md.append(f"# Inference Bench Report\n")
    md.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")

    md.append(f"## Environment\n")
    md.append(f"- GPU: **{meta.get('gpu_name', '?')}** ({meta.get('gpu_total_memory_gb', 0):.1f} GB)")
    md.append(f"- PyTorch: {meta.get('pytorch', '?')}, CUDA: {meta.get('cuda', '?')}")
    md.append(f"- DAT ckpt: `{meta.get('dat_ckpt', '?')}`")
    md.append(f"- DAT layers: {meta.get('dat_layer_indices', '?')}, hr_scale: {meta.get('hr_scale', '?')}")
    md.append(f"- FA backend: {meta.get('fa_backend', '?')}, dtype: {meta.get('dtype', '?')}")
    md.append("")

    # ── prefill ─────────────────────────────────────────────────────────────
    if 'prefill' in payload and payload['prefill']:
        md.append("## Task: Prefill (E2E + Breakdown)\n")
        md.append("Base(R) vs DAT(HD=R, LR=R/hr_scale)，单 batch，单次 prefill。\n")

        rows = []
        for r in payload['prefill']:
            if r.get('oom'):
                rows.append([r['R'], '—', '—', '—', 'OOM', '—', '—', '—', '—'])
                continue
            if 'error' in r:
                rows.append([r['R'], '—', '—', '—', f"ERR: {r['error'][:30]}", '—', '—', '—', '—'])
                continue
            _tq = r.get('t_qwen_e2e_ms')
            _sq = r.get('speedup_fused_vs_qwen')
            rows.append([
                r['R'],
                r['nq_base'], r['nq_lr'],
                f"{_tq:.1f}" if _tq else '—',
                f"{r['t_base_e2e_ms']:.1f}",
                f"{r['t_dat_e2e_ms']:.1f}",
                f"{r['speedup']:.2f}x",
                f"{r.get('t_dat_e2e_fused_ms', 0):.1f}",
                f"{r.get('speedup_fused', 0):.2f}x",
                f"{_sq:.2f}x" if _sq else '—',
                f"{r['mem_base_mb']/1024:.2f} → {r['mem_dat_mb']/1024:.2f}",
            ])
        md.append("### E2E\n")
        md.append(_md_table(rows, ['R', 'Nq Base', 'Nq DAT', 'Qwen ms',
                                   'Base ms', 'DAT ms', 'Speedup',
                                   'DAT-fused ms', 'Speedup(fused)',
                                   'fused vs Qwen',
                                   'Mem GB (Base→DAT)']))

        rows = []
        for r in payload['prefill']:
            if r.get('oom') or 'error' in r:
                continue
            rows.append([
                r['R'],
                f"{r['t_vit_hd_ms']:.1f}",
                f"{r['t_vit_lr_ms']:.1f}",
                f"{r['llm_base_est_ms']:.1f}",
                f"{r['llm_dat_est_ms']:.1f}",
                f"{r['llm_speedup']:.2f}x",
            ])
        md.append("### Breakdown (ms)\n")
        md.append("- ViT_HD = HD ViT 单独耗时 (= Base ViT)\n")
        md.append("- ViT_LR = LR ViT 单独耗时 (DAT 多出的开销)\n")
        md.append("- LLM_Base ≈ Base E2E - ViT_HD\n")
        md.append("- LLM_DAT  ≈ (LR ViT + LLM_DAT) - LR ViT\n")
        md.append(_md_table(rows, ['R', 'ViT_HD', 'ViT_LR',
                                    'LLM_Base', 'LLM_DAT', 'LLM Speedup']))

    # ── batch_decode ─────────────────────────────────────────────────────────
    if 'batch_decode' in payload and payload['batch_decode']:
        md.append("\n## Task: Batch Decode Throughput\n")
        md.append("固定 (R, T)，sweep batch B；TTFT = prefill 耗时，ITL = decode 平均每 token 耗时。\n")

        # 按 (R, T) 分组
        groups = defaultdict(list)
        for r in payload['batch_decode']:
            groups[(r['R'], r['T'])].append(r)

        for (R, T), records in sorted(groups.items()):
            md.append(f"\n### R={R}, decode_len={T}\n")
            # 整理成 per-B 一行，base/dat 并列
            by_b = defaultdict(dict)
            for r in records:
                by_b[r['B']][r['model']] = r
            rows = []
            for B in sorted(by_b.keys()):
                pair = by_b[B]
                qwen = pair.get('qwen', {})
                base = pair.get('base', {}); dat = pair.get('dat', {})
                def fmt(d, k, sfx='', oom_str='OOM'):
                    if not d: return '—'
                    if d.get('oom'): return oom_str
                    if 'error' in d: return 'ERR'
                    v = d.get(k);  return f"{v:.1f}{sfx}" if v is not None else '—'
                def fmt_mem(d):
                    if not d: return '—'
                    if d.get('oom') or 'error' in d: return 'OOM'
                    return f"{d.get('peak_mem_mb', 0)/1024:.2f}"
                row = [
                    B,
                    fmt(qwen, 'ttft_ms'), fmt(base, 'ttft_ms'), fmt(dat, 'ttft_ms'),
                    fmt(qwen, 'itl_ms_per_token'), fmt(base, 'itl_ms_per_token'), fmt(dat, 'itl_ms_per_token'),
                    fmt(qwen, 'tokens_per_sec'), fmt(base, 'tokens_per_sec'), fmt(dat, 'tokens_per_sec'),
                    fmt(base, 'samples_per_sec'), fmt(dat, 'samples_per_sec'),
                    fmt_mem(qwen), fmt_mem(base), fmt_mem(dat),
                ]
                rows.append(row)
            md.append(_md_table(rows, [
                'B',
                'TTFT Qwen', 'TTFT Base', 'TTFT DAT',
                'ITL Qwen', 'ITL Base', 'ITL DAT',
                'tok/s Qwen', 'tok/s Base', 'tok/s DAT',
                'samples/s Base', 'samples/s DAT',
                'Mem Qwen GB', 'Mem Base GB', 'Mem DAT GB',
            ]))

    # ── memory ──────────────────────────────────────────────────────────────
    if 'memory' in payload and payload['memory']:
        md.append("\n## Task: Memory & OOM 临界 (forward-only)\n")
        groups = defaultdict(dict)
        for r in payload['memory']:
            groups[(r['R'], r['B'])][r['model']] = r
        rows = []
        for (R, B) in sorted(groups.keys()):
            pair = groups[(R, B)]
            base = pair.get('base', {}); dat = pair.get('dat', {})
            row = [
                R, B,
                'OOM' if base.get('oom') else f"{base.get('peak_mem_mb', 0)/1024:.2f}" if 'peak_mem_mb' in base else '—',
                'OOM' if dat .get('oom') else f"{dat .get('peak_mem_mb', 0)/1024:.2f}" if 'peak_mem_mb' in dat  else '—',
                f"{base.get('t_ms', 0):.1f}" if 'peak_mem_mb' in base else '—',
                f"{dat .get('t_ms', 0):.1f}" if 'peak_mem_mb' in dat  else '—',
            ]
            rows.append(row)
        md.append(_md_table(rows, ['R', 'B', 'Mem Base GB', 'Mem DAT GB',
                                   't Base ms', 't DAT ms']))

        # OOM 临界总结
        md.append("\n### OOM 临界 batch（forward-only）\n")
        oom_table = defaultdict(lambda: {'base': None, 'dat': None})
        for r in payload['memory']:
            R, B, m = r['R'], r['B'], r['model']
            d = oom_table[R]
            if r.get('oom'):
                d[m + '_first_oom'] = min(d.get(m + '_first_oom', float('inf')), B)
            else:
                d[m + '_max_ok'] = max(d.get(m + '_max_ok', 0), B)
        rows = []
        for R in sorted(oom_table.keys()):
            d = oom_table[R]
            rows.append([
                R,
                d.get('base_max_ok', '—'),
                d.get('base_first_oom', '∞'),
                d.get('dat_max_ok', '—'),
                d.get('dat_first_oom', '∞'),
            ])
        md.append(_md_table(rows, ['R', 'Base max ok', 'Base 1st OOM',
                                   'DAT max ok', 'DAT 1st OOM']))

    # ── layerwise ───────────────────────────────────────────────────────────
    if 'layerwise' in payload and payload['layerwise']:
        md.append("\n## Task: Layerwise (CUDA event hook)\n")
        for r in payload['layerwise']:
            if r.get('oom') or 'error' in r:
                md.append(f"\n### R={r['R']}: SKIPPED ({'OOM' if r.get('oom') else r.get('error', 'err')})\n")
                continue
            md.append(f"\n### R={r['R']}, Nq Base={r['nq_base']}, Nq DAT={r['nq_lr']}\n")
            md.append(f"DAT 层索引: {r['dat_layer_indices']}\n\n")
            md.append(f"- Base total: **{r['total_base_ms']:.1f} ms**\n")
            md.append(f"- DAT  total: **{r['total_dat_ms']:.1f} ms** "
                      f"(diff {r['total_dat_ms']-r['total_base_ms']:+.1f} ms)\n\n")
            rows = []
            for L in r['layers']:
                rows.append([L['layer'], L['type'],
                             f"{L['t_base_ms']:.2f}",
                             f"{L['t_dat_ms']:.2f}",
                             f"{L['diff_ms']:+.2f}"])
            md.append(_md_table(rows, ['layer', 'type', 'Base ms', 'DAT ms', 'diff ms']))

    # ── vit_paths ───────────────────────────────────────────────────────────
    if 'vit_paths' in payload and payload['vit_paths']:
        md.append("\n## Task: ViT Paths (5-way)\n")
        rows = []
        for r in payload['vit_paths']:
            if r.get('oom') or 'error' in r:
                rows.append([r['R'], 'OOM', '—', '—', '—', '—', '—'])
                continue
            rows.append([
                r['R'],
                f"{r['t_dat_sep_ms']:.1f}",
                f"{r['t_dat_fused_ms']:.1f}",
                f"{r['t_dat_shared_ms']:.1f}",
                f"{r['t_fair_hd_ms']:.1f}",
                f"{r['t_lr_only_ms']:.1f}",
                f"{r['t_hd_enc_ms']:.1f}",
            ])
        md.append(_md_table(rows, ['R', 'DAT-Sep', 'DAT-Fused', 'DAT-Shared',
                                    'Fair-HD', 'LR-only', 'HD ViT only']))

    # ── Takeaways ───────────────────────────────────────────────────────────
    md.append("\n## Takeaways\n")
    if 'prefill' in payload:
        speedups = [r['speedup'] for r in payload['prefill']
                    if not r.get('oom') and 'speedup' in r]
        if speedups:
            md.append(f"- **Prefill Speedup**: avg {sum(speedups)/len(speedups):.2f}x, "
                      f"max {max(speedups):.2f}x\n")
    if 'batch_decode' in payload:
        # 计算各 B 下 DAT/Base throughput 比
        ratios = []
        groups = defaultdict(dict)
        for r in payload['batch_decode']:
            if 'tokens_per_sec' in r:
                groups[(r['R'], r['B'], r['T'])][r['model']] = r['tokens_per_sec']
        for k, v in groups.items():
            if 'base' in v and 'dat' in v and v['base'] > 0:
                ratios.append(v['dat'] / v['base'])
        if ratios:
            md.append(f"- **Decode Throughput Ratio (DAT/Base)**: "
                      f"avg {sum(ratios)/len(ratios):.2f}x, max {max(ratios):.2f}x\n")
    if 'memory' in payload:
        # 显存节省
        savings = []
        groups = defaultdict(dict)
        for r in payload['memory']:
            if 'peak_mem_mb' in r:
                groups[(r['R'], r['B'])][r['model']] = r['peak_mem_mb']
        for k, v in groups.items():
            if 'base' in v and 'dat' in v and v['base'] > 0:
                savings.append(1 - v['dat'] / v['base'])
        if savings:
            md.append(f"- **Memory Saving (1 - DAT/Base)**: "
                      f"avg {sum(savings)/len(savings)*100:.1f}%, "
                      f"max {max(savings)*100:.1f}%\n")

    return '\n'.join(md)


# ════════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL DAT 统一推理基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-model",  type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dat-ckpt",    type=str, default=DEFAULT_DAT_CKPT)
    parser.add_argument("--vstar-dir",   type=str, default=DEFAULT_VSTAR_DIR)
    parser.add_argument("--vstar-jsonl", type=str, default=DEFAULT_VSTAR_JSONL)
    parser.add_argument("--synthetic",   action="store_true",
                        help="使用合成图像（不依赖 vstar 数据集）")
    parser.add_argument("--no-native",   action="store_true",
                        help="不加载原生 Qwen2.5-VL 对照基线（省显存/时间）")
    parser.add_argument("--question",    type=str, default=DEFAULT_QUESTION)
    parser.add_argument("--hd-early-k",  type=int, default=0,
                        help="HD ViT 早退: 只跑前 k 个 vision block (0=关)。"
                             "只影响 DAT 的 separate HD 路径; 与 fused/shared 不兼容")
    parser.add_argument("--attn-impl",   type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"],
                        help="DAT 与原生 Qwen 统一的 attention backend (含 ViT)。"
                             "sdpa 的 ViT window-attn 比 fa2 慢 ~2x, 默认 fa2")

    parser.add_argument("--tasks", nargs="+",
                        default=DEFAULT_TASKS,
                        help=f"待跑的任务，可选: {ALL_TASKS} 或 'all'。默认: {DEFAULT_TASKS}")

    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=DEFAULT_BATCH_SIZES,
                        help="batch_decode / memory 用的 batch 列表")
    parser.add_argument("--decode-lens", type=int, nargs="+",
                        default=DEFAULT_DECODE_LENS,
                        help="batch_decode 用的 decode 长度列表")
    parser.add_argument("--layerwise-r", type=int, nargs="+",
                        default=DEFAULT_LAYERWISE_R,
                        help="layerwise 任务用的分辨率（默认只测一个 R 减少开销）")

    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters",  type=int, default=DEFAULT_ITERS)
    parser.add_argument("--layerwise-iters", type=int, default=5)

    parser.add_argument("--output-dir", type=str, default="bench_outputs")
    parser.add_argument("--tag",        type=str, default=None,
                        help="输出文件名前缀。默认用时间戳")
    args = parser.parse_args()

    # 解析 tasks
    if 'all' in args.tasks:
        tasks = ALL_TASKS
    else:
        tasks = [t for t in args.tasks if t in ALL_TASKS]
        bad = [t for t in args.tasks if t not in ALL_TASKS and t != 'all']
        if bad:
            print(f"WARN: unknown tasks ignored: {bad}")
    if not tasks:
        print("No tasks selected, exit.")
        return

    # ── 输出文件准备 ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or time.strftime('%Y%m%d_%H%M%S')
    out_json = os.path.join(args.output_dir, f"bench_{tag}.json")
    out_md   = os.path.join(args.output_dir, f"bench_{tag}.md")

    # ── 模型加载 ────────────────────────────────────────────────────────────
    from transformers import AutoConfig, AutoProcessor
    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        Qwen2_5_VLDATForConditionalGeneration, convert_qwen2_5vl_to_dat, _FA_BACKEND,
    )

    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.version.cuda}")
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"FA backend: {_FA_BACKEND}")
    print(f"DAT ckpt : {args.dat_ckpt or '(none — 从 base 构建, DAT 模块随机初始化)'}")
    print(f"Tasks    : {tasks}")
    print()

    processor = AutoProcessor.from_pretrained(args.dat_ckpt or args.base_model)

    if args.dat_ckpt:
        print(f"Loading DAT model from ckpt (attn={args.attn_impl}) …")
        dat_model = Qwen2_5_VLDATForConditionalGeneration.from_pretrained(
            args.dat_ckpt, torch_dtype=DTYPE, device_map={"": 0},
            attn_implementation=args.attn_impl,
        ).eval()
    else:
        # 测速模式: base 权重 + 随机初始化 DAT 模块 (耗时与训练权重一致)
        base_cfg = AutoConfig.from_pretrained(args.base_model)
        n_layers = getattr(base_cfg, 'num_hidden_layers', None) or \
                   base_cfg.text_config.num_hidden_layers
        dat_extra_args = dict(DEFAULT_DAT_EXTRA_ARGS)
        dat_extra_args['layers'] = ''.join(
            'D' if i % 6 == 0 else 'L' for i in range(n_layers))
        print(f"Building DAT from base ({n_layers} layers, "
              f"pattern={dat_extra_args['layers']}, attn={args.attn_impl}) …")
        dat_model = convert_qwen2_5vl_to_dat(
            args.base_model, dat_extra_args, torch_dtype=DTYPE,
            attn_implementation=args.attn_impl,
        ).to(DEVICE).eval()

    qwen_model = None
    if not args.no_native:
        from transformers import Qwen2_5_VLForConditionalGeneration
        print(f"Loading native Qwen2.5-VL (官方类, 对照基线, attn={args.attn_impl}) …")
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=DTYPE, device_map={"": 0},
            attn_implementation=args.attn_impl,
        ).eval()

    # 关 padding warning（generate 时左 padding，processor 默认右 padding）
    processor.tokenizer.padding_side = "left"

    dat_ea     = dat_model.config.dat_extra_args
    hr_scale   = dat_ea.get('hr_scale', 3)
    layers_str = dat_ea.get('layers', '')
    dat_layer_indices = [i for i, c in enumerate(layers_str) if c == 'D']

    # 强制走 separate ViT 路径（公平对比，避免之前残留 flag）
    _set_vit_path(dat_model, 'separate')

    # HD 早退 (只对 separate HD 路径生效; prefill 任务里的 DAT-fused 列不受影响,
    # 早退开启时请只看非 fused 的 DAT 数字)
    dat_ea['hd_early_exit_k'] = args.hd_early_k
    if args.hd_early_k:
        print(f"  HD early exit: 前 {args.hd_early_k} 个 vision block")

    print(f"  hr_scale={hr_scale}, DAT layers={dat_layer_indices}")
    print(f"  GPU mem after load: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

    # ── 准备图像 ────────────────────────────────────────────────────────────
    if args.synthetic or not os.path.exists(args.vstar_jsonl):
        max_R = max(args.resolutions)
        print(f"\nUsing synthetic image {max_R}×{max_R}")
        img = make_synthetic_image(max_R)
    else:
        print(f"\nLoading first vstar image …")
        img = None
        with open(args.vstar_jsonl) as f:
            for line in f:
                item = json.loads(line.strip())
                ip = os.path.join(args.vstar_dir, item['image'])
                if os.path.exists(ip):
                    try:
                        img = Image.open(ip).convert("RGB")
                        print(f"  loaded {ip}")
                        break
                    except Exception:
                        continue
        if img is None:
            max_R = max(args.resolutions)
            print(f"  no vstar image found, fallback to synthetic {max_R}×{max_R}")
            img = make_synthetic_image(max_R)

    # ── 元信息 ──────────────────────────────────────────────────────────────
    metadata = {
        **gpu_info(),
        'fa_backend': _FA_BACKEND,
        'dtype':      str(DTYPE),
        'dat_ckpt':   args.dat_ckpt,
        'hr_scale':   hr_scale,
        'attn_impl':  args.attn_impl,
        'hd_early_exit_k': args.hd_early_k,
        'dat_layer_indices': dat_layer_indices,
        'tasks':      tasks,
        'resolutions': args.resolutions,
        'batch_sizes': args.batch_sizes,
        'decode_lens': args.decode_lens,
        'warmup':     args.warmup,
        'iters':      args.iters,
        'use_synthetic': args.synthetic or not os.path.exists(args.vstar_jsonl),
        'timestamp':  time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    payload = {'metadata': metadata}

    def save():
        """增量保存。"""
        write_json(out_json, payload)

    save()

    # ── 跑各任务 ────────────────────────────────────────────────────────────
    if 'prefill' in tasks:
        print(f"\n{'═'*88}\n[Task: prefill] E2E + Breakdown\n{'═'*88}")
        payload['prefill'] = run_prefill_sweep(
            dat_model, processor, img,
            resolutions=args.resolutions, hr_scale=hr_scale,
            warmup=args.warmup, iters=args.iters, save_cb=save,
            qwen_model=qwen_model,
        )

    if 'batch_decode' in tasks:
        print(f"\n{'═'*88}\n[Task: batch_decode] Decode throughput sweep\n{'═'*88}")
        payload['batch_decode'] = run_batch_decode_sweep(
            dat_model, processor, img,
            resolutions=args.resolutions,
            batch_sizes=args.batch_sizes,
            decode_lens=args.decode_lens,
            hr_scale=hr_scale,
            warmup=args.warmup, iters=args.iters, save_cb=save,
            qwen_model=qwen_model,
        )

    if 'memory' in tasks:
        print(f"\n{'═'*88}\n[Task: memory] Forward-only memory sweep\n{'═'*88}")
        payload['memory'] = run_memory_sweep(
            dat_model, processor, img,
            resolutions=args.resolutions,
            batch_sizes=args.batch_sizes,
            hr_scale=hr_scale,
            warmup=args.warmup, iters=args.iters, save_cb=save,
        )

    if 'layerwise' in tasks:
        print(f"\n{'═'*88}\n[Task: layerwise] Per-layer CUDA event timing\n{'═'*88}")
        payload['layerwise'] = run_layerwise_sweep(
            dat_model, processor, img,
            resolutions=args.layerwise_r, hr_scale=hr_scale,
            dat_layer_indices=dat_layer_indices,
            warmup=args.warmup, n_iters=args.layerwise_iters, save_cb=save,
        )

    if 'vit_paths' in tasks:
        print(f"\n{'═'*88}\n[Task: vit_paths] 5-way ViT comparison\n{'═'*88}")
        payload['vit_paths'] = run_vit_paths_sweep(
            dat_model, processor, img,
            resolutions=args.resolutions, hr_scale=hr_scale,
            warmup=args.warmup, iters=args.iters, save_cb=save,
        )

    # ── 写输出 ──────────────────────────────────────────────────────────────
    save()
    md = render_markdown(payload)
    with open(out_md, 'w') as f:
        f.write(md)

    print(f"\n{'═'*88}")
    print(f"✓ Done.")
    print(f"  JSON: {out_json}")
    print(f"  MD:   {out_md}")
    print(f"{'═'*88}")


if __name__ == "__main__":
    main()
