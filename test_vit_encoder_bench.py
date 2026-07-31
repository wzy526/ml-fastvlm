#!/usr/bin/env python3
"""
test_vit_encoder_bench.py — Qwen2.5-VL ViT 编码器专项测速 (无数据依赖)。

背景
────
DAT 双路编码 = ViT(LR, R/hr_scale) + ViT(HD, R)，比 Base 单次 ViT(R) 多出
LR 一趟 + 两次调用的固定开销。本脚本把这部分开销测清楚，为 ViT 效率优化
提供基线。全部输入为合成随机张量，不需要 ckpt / 数据 / 网络。

任务
────
  paths      — 每个 R 上对比:
                 t_hd        单次 ViT(R)            (Base 路径)
                 t_lr        单次 ViT(R/hr_scale)
                 t_two_pass  ViT(LR) + ViT(HD) 串行  (DAT 现状)
                 t_fused     LR+HD 拼一个 batch 单次调用 (DAT 可能的改进)
                 t_streams   LR/HD 双 CUDA stream 并发 (streaming 调用改进)
               并给出 overhead = two_pass - (hd + lr)，dat/base 比值。
  breakdown  — 逐模块 CUDA event 计时: patch_embed / 每个 block
               (区分 window-attn vs full-attn) / merger，
               以及 unaccounted (= 总时间 - 各模块之和，
               主要是 get_window_index 等 CPU 侧准备开销)。
  early_exit — HD 早退分层测速: 只跑前 k 个 block (临时截断 visual.blocks,
               merger/窗口重排不变), k 扫 --early-ks。每个 (R, k) 给出:
                 t_actual   实测截断耗时
                 t_pred     逐 block CUDA event 计时累加的预测值
                            (= 固定开销 + sum(block[:k]))
                 flops_frac 前 k 个 block 的 FLOPs 占全深度比例
               t_actual vs t_pred 的差 = 截断额外省掉的 CPU 发射开销;
               t_actual/t_full vs flops_frac 的差 = FLOPs 反映不了的部分。

用法
────
  python test_vit_encoder_bench.py \
      --model-path /workspace/model_cache/Qwen2.5-VL-3B-Instruct \
      --resolutions 672 1008 1344 2016 2688 --hr-scale 3

  # 不加载权重 (纯 config 随机初始化, 速度与真权重一致)
  python test_vit_encoder_bench.py --random-init
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import contextlib
import json
import statistics
import time
from datetime import datetime

import torch

DEVICE = "cuda"
DTYPE = torch.bfloat16

DEFAULT_MODEL = os.path.join(
    os.environ.get("MODEL_CACHE", "/workspace/model_cache"),
    "Qwen2.5-VL-3B-Instruct")
# R 需同时是 28 (patch14 x merge2) 和 28*hr_scale 的倍数
DEFAULT_RESOLUTIONS = [672, 1008, 1344, 2016, 2688]


# ────────────────────────────────────────────────────────────────────
# 模型加载: 只要视觉塔
# ────────────────────────────────────────────────────────────────────

def build_visual(model_path: str, random_init: bool):
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    cfg = AutoConfig.from_pretrained(model_path)
    vcfg = cfg.vision_config

    if random_init:
        vcfg._attn_implementation = "flash_attention_2"
        visual = Qwen2_5_VisionTransformerPretrainedModel(vcfg)
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration

        full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=DTYPE, attn_implementation="flash_attention_2"
        )
        # transformers 5.x: model.visual; 兼容旧版扁平布局
        visual = full.model.visual if hasattr(full.model, "visual") else full.visual

    visual = visual.to(device=DEVICE, dtype=DTYPE).eval()
    torch.cuda.empty_cache()
    return visual, vcfg


# ────────────────────────────────────────────────────────────────────
# 合成输入: 直接构造 patch 序列, 绕过 processor
# ────────────────────────────────────────────────────────────────────

def make_inputs(R: int, vcfg, batch: int = 1):
    """R×R 图像对应的 (pixel_values, grid_thw)。"""
    ps = vcfg.patch_size                     # 14
    tps = vcfg.temporal_patch_size           # 2
    h = w = R // ps
    assert h % 2 == 0, f"R={R}: patch 网格 {h} 须为偶数 (merge_size=2)"
    dim = vcfg.in_channels * tps * ps * ps   # 3*2*14*14 = 1176
    pv = torch.randn(batch * h * w, dim, device=DEVICE, dtype=DTYPE)
    thw = torch.tensor([[1, h, w]] * batch, device=DEVICE, dtype=torch.long)
    return pv, thw


def n_vis_tokens(thw) -> int:
    return int((thw[:, 0] * (thw[:, 1] // 2) * (thw[:, 2] // 2)).sum())


# ────────────────────────────────────────────────────────────────────
# 计时
# ────────────────────────────────────────────────────────────────────

def bench(fn, warmup: int, iters: int):
    """返回 (mean_ms, std_ms)。"""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(ts), (statistics.stdev(ts) if len(ts) > 1 else 0.0)


# ────────────────────────────────────────────────────────────────────
# Task: paths
# ────────────────────────────────────────────────────────────────────

def task_paths(visual, vcfg, R: int, hr_scale: int, batch: int,
               warmup: int, iters: int):
    lr = R // hr_scale
    pv_hd, thw_hd = make_inputs(R, vcfg, batch)
    pv_lr, thw_lr = make_inputs(lr, vcfg, batch)
    pv_cat = torch.cat([pv_lr, pv_hd], dim=0)
    thw_cat = torch.cat([thw_lr, thw_hd], dim=0)

    t_hd, s_hd = bench(lambda: visual(pv_hd, grid_thw=thw_hd), warmup, iters)
    t_lr, s_lr = bench(lambda: visual(pv_lr, grid_thw=thw_lr), warmup, iters)

    def two_pass():
        visual(pv_lr, grid_thw=thw_lr)
        visual(pv_hd, grid_thw=thw_hd)

    t_two, s_two = bench(two_pass, warmup, iters)
    t_fused, s_fused = bench(lambda: visual(pv_cat, grid_thw=thw_cat), warmup, iters)

    # streaming: LR 放侧 stream, 与默认 stream 上的 HD 并发, LR 时间被 HD 隐藏
    side = torch.cuda.Stream()

    def streamed():
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            visual(pv_lr, grid_thw=thw_lr)
        visual(pv_hd, grid_thw=thw_hd)
        torch.cuda.current_stream().wait_stream(side)

    t_str, s_str = bench(streamed, warmup, iters)

    return {
        "R": R, "LR": lr, "batch": batch,
        "hd_tokens": n_vis_tokens(thw_hd), "lr_tokens": n_vis_tokens(thw_lr),
        "t_hd_ms": t_hd, "t_hd_std": s_hd,
        "t_lr_ms": t_lr, "t_lr_std": s_lr,
        "t_two_pass_ms": t_two, "t_two_pass_std": s_two,
        "t_fused_ms": t_fused, "t_fused_std": s_fused,
        "t_streams_ms": t_str, "t_streams_std": s_str,
        "overhead_ms": t_two - (t_hd + t_lr),
        "two_pass_over_hd": t_two / t_hd if t_hd > 0 else 0,
        "fused_over_hd": t_fused / t_hd if t_hd > 0 else 0,
        "streams_over_hd": t_str / t_hd if t_hd > 0 else 0,
    }


# ────────────────────────────────────────────────────────────────────
# Task: breakdown (hook + CUDA event)
# ────────────────────────────────────────────────────────────────────

def task_breakdown(visual, vcfg, R: int, warmup: int, iters: int):
    pv, thw = make_inputs(R, vcfg, batch=1)
    fullatt = set(getattr(vcfg, "fullatt_block_indexes", []) or [])

    mods = {"patch_embed": visual.patch_embed, "merger": visual.merger}
    for i, blk in enumerate(visual.blocks):
        mods[f"block_{i}"] = blk

    starts, pairs, handles = {}, {name: [] for name in mods}, []

    def mk_pre(name):
        def pre(_m, _args):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            starts[name] = e
        return pre

    def mk_post(name):
        def post(_m, _args, _out):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            pairs[name].append((starts[name], e))
        return post

    for name, m in mods.items():
        handles.append(m.register_forward_pre_hook(mk_pre(name)))
        handles.append(m.register_forward_hook(mk_post(name)))

    try:
        with torch.no_grad():
            for _ in range(warmup):
                visual(pv, grid_thw=thw)
            torch.cuda.synchronize()
            for name in pairs:
                pairs[name].clear()

            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                visual(pv, grid_thw=thw)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
    finally:
        for h in handles:
            h.remove()

    per_mod = {name: sum(s.elapsed_time(e) for s, e in ps) / iters
               for name, ps in pairs.items()}

    t_window = sum(v for k, v in per_mod.items()
                   if k.startswith("block_") and int(k.split("_")[1]) not in fullatt)
    t_full = sum(v for k, v in per_mod.items()
                 if k.startswith("block_") and int(k.split("_")[1]) in fullatt)
    total = statistics.mean(ts)
    accounted = per_mod["patch_embed"] + per_mod["merger"] + t_window + t_full

    return {
        "R": R, "total_ms": total,
        "patch_embed_ms": per_mod["patch_embed"],
        "window_blocks_ms": t_window,
        "full_blocks_ms": t_full,
        "merger_ms": per_mod["merger"],
        "unaccounted_ms": total - accounted,
        "n_window_blocks": len(visual.blocks) - len(fullatt),
        "n_full_blocks": len(fullatt),
        "per_block_ms": {k: v for k, v in per_mod.items() if k.startswith("block_")},
    }


# ────────────────────────────────────────────────────────────────────
# Task: early_exit (只跑前 k 个 block)
# ────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def truncated_blocks(visual, k: int):
    """临时把 visual.blocks 截断为前 k 个 (窗口重排/merger 不变)。"""
    full = visual.blocks
    if 0 < k < len(full):
        visual.blocks = full[:k]
    try:
        yield
    finally:
        visual.blocks = full


def vit_prefix_flops(R: int, k: int, vcfg) -> float:
    """前 k 个 block 的 ViT FLOPs (含 patch_embed + merger, window ctx=8x8=64)。"""
    d, inter = vcfg.hidden_size, vcfg.intermediate_size
    depth = vcfg.depth
    fullatt = set(getattr(vcfg, "fullatt_block_indexes", []) or [])
    N = (R // vcfg.patch_size) ** 2
    win_ctx = (vcfg.window_size // vcfg.patch_size) ** 2

    f = 2 * N * (vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2) * d
    for i in range(min(k, depth)):
        f += 2 * N * (4 * d ** 2 + 3 * d * inter)                    # qkv/proj + SwiGLU
        f += 4 * N * (N if i in fullatt else win_ctx) * d            # attn matmul
    merge_dim = d * vcfg.spatial_merge_size ** 2
    f += 2 * (N // 4) * (merge_dim ** 2 + merge_dim * vcfg.out_hidden_size)
    return f


def task_early_exit(visual, vcfg, R: int, ks, warmup: int, iters: int):
    """实测截断 t(k) + 逐 block 计时预测 + FLOPs 比例，三方对照。"""
    depth = len(visual.blocks)

    # 1) 逐 block 计时 (复用 breakdown 的 hook 逻辑; 仅用于预测,
    #    其 total_ms 含 hook 自身的 CPU 开销, 不作为 t_full)
    brk = task_breakdown(visual, vcfg, R, warmup, iters)
    per_block = [brk["per_block_ms"][f"block_{i}"] for i in range(depth)]
    # 固定开销 = patch_embed + merger + unaccounted (CPU 侧准备/发射等)
    fixed = brk["patch_embed_ms"] + brk["merger_ms"] + brk["unaccounted_ms"]
    flops_full = vit_prefix_flops(R, depth, vcfg)

    # 2) 实测: 全深度基准 (无 hook) + 各 k 截断
    pv, thw = make_inputs(R, vcfg, batch=1)
    t_full, _ = bench(lambda: visual(pv, grid_thw=thw), warmup, iters)
    rows = []
    for k in ks:
        if not 0 < k <= depth:
            continue
        if k == depth:
            t_k, s_k = t_full, 0.0
        else:
            with truncated_blocks(visual, k):
                t_k, s_k = bench(lambda: visual(pv, grid_thw=thw), warmup, iters)
        t_pred = fixed + sum(per_block[:k])
        rows.append({
            "R": R, "k": k, "depth": depth,
            "t_ms": t_k, "t_std": s_k,
            "t_pred_ms": t_pred,
            "t_full_ms": t_full,
            "ratio_vs_full": t_k / t_full if t_full > 0 else 0,
            "flops_frac": vit_prefix_flops(R, k, vcfg) / flops_full,
            "launch_saved_ms": t_pred - t_k,   # >0: 截断额外省的发射开销
        })
    return rows, brk


# ────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--random-init", action="store_true",
                   help="不加载权重, 纯 config 初始化 (测速结果一致)")
    p.add_argument("--tasks", nargs="+", default=["paths", "breakdown"],
                   choices=["paths", "breakdown", "early_exit"])
    p.add_argument("--resolutions", type=int, nargs="+", default=DEFAULT_RESOLUTIONS)
    p.add_argument("--hr-scale", type=int, default=3)
    p.add_argument("--early-ks", type=int, nargs="+",
                   default=[4, 8, 12, 16, 20, 24, 28, 32],
                   help="early_exit 任务的 k 列表 (跑前 k 个 block)")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--output-dir", default="bench_outputs")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[load] {args.model_path}  random_init={args.random_init}")
    visual, vcfg = build_visual(args.model_path, args.random_init)
    n_params = sum(x.numel() for x in visual.parameters()) / 1e6
    print(f"[info] ViT params={n_params:.0f}M  depth={len(visual.blocks)}"
          f"  fullatt_blocks={getattr(vcfg, 'fullatt_block_indexes', None)}"
          f"  gpu={torch.cuda.get_device_name(0)}")

    unit = 28 * args.hr_scale
    resolutions = []
    for R in args.resolutions:
        if R % unit == 0:
            resolutions.append(R)
        else:
            print(f"  [skip] R={R} 非 {unit} 的倍数 (28 x hr_scale)")

    payload = {
        "meta": {
            "model": args.model_path, "random_init": args.random_init,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "dtype": str(DTYPE),
            "hr_scale": args.hr_scale, "warmup": args.warmup, "iters": args.iters,
            "time": datetime.now().isoformat(),
        }
    }

    if "paths" in args.tasks:
        print(f"\n{'=' * 100}\n[paths] Base 单趟 vs DAT 双趟 vs Fused 拼批  (hr_scale={args.hr_scale})\n{'=' * 100}")
        rows = []
        for B in args.batch_sizes:
            for R in resolutions:
                try:
                    r = task_paths(visual, vcfg, R, args.hr_scale, B,
                                   args.warmup, args.iters)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  R={R:>5} B={B:<2} OOM, skip")
                    rows.append({"R": R, "batch": B, "oom": True})
                    continue
                rows.append(r)
                print(f"  R={r['R']:>5} B={B:<2} tokens(HD/LR)={r['hd_tokens']}/{r['lr_tokens']}"
                      f"  HD={r['t_hd_ms']:7.2f}  LR={r['t_lr_ms']:6.2f}"
                      f"  2pass={r['t_two_pass_ms']:7.2f} ({r['two_pass_over_hd']:.3f}x)"
                      f"  fused={r['t_fused_ms']:7.2f} ({r['fused_over_hd']:.3f}x)"
                      f"  streams={r['t_streams_ms']:7.2f} ({r['streams_over_hd']:.3f}x)")
        payload["paths"] = rows

    if "breakdown" in args.tasks:
        print(f"\n{'=' * 100}\n[breakdown] 逐模块分解 (单趟 ViT)\n{'=' * 100}")
        rows = []
        for R in resolutions:
            try:
                r = task_breakdown(visual, vcfg, R, args.warmup, args.iters)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  R={R:>5} OOM, skip")
                rows.append({"R": R, "oom": True})
                continue
            rows.append(r)
            print(f"  R={r['R']:>5} total={r['total_ms']:7.2f}"
                  f"  patch_embed={r['patch_embed_ms']:5.2f}"
                  f"  window x{r['n_window_blocks']}={r['window_blocks_ms']:7.2f}"
                  f"  full x{r['n_full_blocks']}={r['full_blocks_ms']:6.2f}"
                  f"  merger={r['merger_ms']:5.2f}"
                  f"  unaccounted={r['unaccounted_ms']:+.2f}")
        payload["breakdown"] = rows

    if "early_exit" in args.tasks:
        print(f"\n{'=' * 100}\n[early_exit] 分层测速: 实测截断 vs 逐层预测 vs FLOPs (ks={args.early_ks})\n{'=' * 100}")
        rows, brks = [], []
        for R in resolutions:
            try:
                rs, brk = task_early_exit(visual, vcfg, R, args.early_ks,
                                          args.warmup, args.iters)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  R={R:>5} OOM, skip")
                rows.append({"R": R, "oom": True})
                continue
            rows.extend(rs)
            brks.append(brk)
            if not rs:
                continue
            fixed = brk["patch_embed_ms"] + brk["merger_ms"] + brk["unaccounted_ms"]
            print(f"\n  R={R:>5}  full={rs[0]['t_full_ms']:7.2f}ms"
                  f"  (固定开销 patch+merger+unacc = {fixed:.2f}ms)")
            print(f"    {'k':>4} {'实测ms':>9} {'预测ms':>9} {'实测/full':>9} "
                  f"{'FLOPs占比':>9} {'省的发射ms':>10}")
            for r in rs:
                print(f"    {r['k']:>4} {r['t_ms']:>9.2f} {r['t_pred_ms']:>9.2f} "
                      f"{r['ratio_vs_full']:>8.2f}x {r['flops_frac']:>8.2f}x "
                      f"{r['launch_saved_ms']:>10.2f}")
        payload["early_exit"] = rows
        payload["early_exit_breakdown"] = brks

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%m%d_%H%M%S")
    out = os.path.join(args.output_dir, f"vit_bench_{tag}.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
