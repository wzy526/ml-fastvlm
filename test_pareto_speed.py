#!/usr/bin/env python3
"""
test_pareto_speed.py  (v2)

固定分辨率 Pareto 速度测试：DAT(HD=R, LR=R/3) vs Base(R)

三级测量：
  ① E2E Prefill   : Base(R) vs DAT(HD=R) 全流程延迟
  ② 分解 Breakdown : ViT(HD) / ViT(LR) / LLM_Base / LLM_DAT 各自耗时
  ③ Layerwise      : 每个 decoder layer 的 CUDA event 计时（Base vs DAT）

实验设计：
  Base(R)   = DAT 模型关闭 HD 路，pixel_values_hd=None，全因果注意力
  DAT(HD=R) = HD=R，LR=R/hr_scale，稀疏 HD cross-attn + 短 LLM 序列

分解逻辑：
  t_vit_hd        : _generate_hd_features(pv_hd)          → ViT 在 R 分辨率
  t_vit_lr        : _generate_hd_features(pv_lr)          → ViT 在 R/3 分辨率
  t_llm_dat_pre   : forward(pv_lr, image_hd_features=pre) → LR ViT + LLM_DAT（HD已预算）
  LLM_DAT  ≈ t_llm_dat_pre - t_vit_lr
  LLM_Base ≈ t_base_e2e    - t_vit_hd

默认测试分辨率 (全部为 84=28×3 的倍数):
  R ∈ {336, 504, 672, 1008, 1344, 2016}

用法:
  python test_pareto_speed.py [options]
  python test_pareto_speed.py --synthetic --warmup 5 --iters 20
  python test_pareto_speed.py --no-layerwise   # 跳过逐层测试（更快）
  python test_pareto_speed.py --layerwise-only-r 1344 2016
"""

import os, sys, time, json, argparse
from collections import defaultdict

import torch
from PIL import Image

# ─── 默认路径 ────────────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL  = '/cluster/data3/wzy/vldat_experiments/Qwen2.5-VL-3B-Instruct/Qwen/Qwen2___5-VL-3B-Instruct'
DEFAULT_DAT_CKPT    = '/cluster/data3/wzy/vldat_experiments/checkpoint-3907'
DEFAULT_VSTAR_DIR   = '/cluster/data3/wzy/sft_data_prev/sft_data/vstar_bench'
DEFAULT_VSTAR_JSONL = os.path.join(DEFAULT_VSTAR_DIR, 'test_questions.jsonl')

# ─── 模型常量 ────────────────────────────────────────────────────────────────
PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR     = PATCH_SIZE * MERGE_SIZE   # 28
HR_SCALE   = 3   # 从 checkpoint config 读取后覆盖

DTYPE   = torch.bfloat16
DEVICE  = 'cuda'

DEFAULT_RESOLUTIONS = [336, 504, 672, 1008, 1344, 2016]
DEFAULT_WARMUP      = 3
DEFAULT_ITERS       = 20
DEFAULT_MAX_SAMPLES = 1
LAYERWISE_ITERS     = 5

QUESTION = "Describe this image in detail."


# ════════════════════════════════════════════════════════════════════════════════
# 基础工具
# ════════════════════════════════════════════════════════════════════════════════

def visual_tokens(inputs) -> int:
    thw = inputs["image_grid_thw"][0]
    t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
    return (h // MERGE_SIZE) * (w // MERGE_SIZE) * t


def benchmark_fn(fn, warmup: int = 0, iters: int = 20) -> float:
    """返回平均耗时 (ms/iter)，含 CUDA 同步。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def make_text(processor, img: Image.Image) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": QUESTION},
    ]}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ════════════════════════════════════════════════════════════════════════════════
# 图像预处理
# ════════════════════════════════════════════════════════════════════════════════

def process_base(processor, img, text, R, device=DEVICE):
    """Base(R): min=max=R² 像素，全因果注意力。"""
    inp = processor(images=[img], text=[text], return_tensors="pt",
                    padding=False, min_pixels=R*R, max_pixels=R*R)
    return {k: v.to(device) for k, v in inp.items()}


def process_dat(processor, img, text, R, hr_scale, device=DEVICE):
    """
    DAT 双路:
      inp_lr  : LR 分辨率 (R/hr_scale)²，含完整问题文本
      inp_hd  : HD 分辨率 R²，仅用于提取 pixel_values/image_grid_thw
    """
    lr = R // hr_scale
    inp_lr = processor(images=[img], text=[text], return_tensors="pt",
                       padding=False, min_pixels=lr*lr, max_pixels=lr*lr)
    inp_hd = processor(images=[img], text=["<|im_start|>"], return_tensors="pt",
                       padding=False, min_pixels=R*R, max_pixels=R*R)
    return (
        {k: v.to(device) for k, v in inp_lr.items()},
        {k: v.to(device) for k, v in inp_hd.items()},
    )


# ════════════════════════════════════════════════════════════════════════════════
# ① E2E Prefill + ② 分解 Breakdown
# ════════════════════════════════════════════════════════════════════════════════

def measure_one_resolution(dat_model, processor, img, text, R, hr_scale,
                            warmup, iters):
    """
    对单张图在分辨率 R 上测量所有分解时间，返回 dict。

    分解：
      t_base_e2e     : Base(R) 全流程 (ViT_hd + LLM_base)
      t_dat_e2e      : DAT(HD=R) 全流程 (ViT_lr + ViT_hd + LLM_dat)
      t_vit_hd       : ViT 在 R 分辨率（= Base ViT）
      t_vit_lr       : ViT 在 R/hr_scale 分辨率
      t_llm_dat_pre  : LR ViT + LLM_DAT（HD 已预算，跳过 HD ViT）
      llm_base       : ≈ t_base_e2e - t_vit_hd   (估算)
      llm_dat        : ≈ t_llm_dat_pre - t_vit_lr (估算)
    """
    lr = R // hr_scale

    inp_base        = process_base(processor, img, text, R)
    inp_lr, inp_hd  = process_dat(processor, img, text, R, hr_scale)

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

    # 预计算 HD features（用于 LLM_DAT pre 测量，排除 HD ViT）
    with torch.no_grad():
        hd_feats = dat_model._generate_hd_features(pv_hd, thw_hd)

    # ── 定义各路 forward ─────────────────────────────────────────────────────

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
                      image_hd_features=hd_feats,
                      use_cache=False)

    # ── 预热（所有路径都走一遍）─────────────────────────────────────────────
    for _ in range(warmup):
        fwd_base_e2e(); fwd_dat_e2e(); fwd_vit_hd(); fwd_vit_lr(); fwd_llm_dat_pre()
    torch.cuda.synchronize()

    # ── 计时 ──────────────────────────────────────────────────────────────────
    t_base_e2e    = benchmark_fn(fwd_base_e2e,   iters=iters)
    t_dat_e2e     = benchmark_fn(fwd_dat_e2e,    iters=iters)
    t_vit_hd      = benchmark_fn(fwd_vit_hd,     iters=iters)
    t_vit_lr      = benchmark_fn(fwd_vit_lr,     iters=iters)
    t_llm_dat_pre = benchmark_fn(fwd_llm_dat_pre, iters=iters)

    llm_base = max(0.0, t_base_e2e    - t_vit_hd)   # 估算
    llm_dat  = max(0.0, t_llm_dat_pre - t_vit_lr)   # 估算

    return {
        'R': R, 'LR': lr,
        'HD_tokens': n_vis_h, 'LR_tokens': n_vis_l,
        'nq_base': nq_base, 'nq_lr': nq_lr,
        # E2E
        't_base_e2e_ms':    t_base_e2e,
        't_dat_e2e_ms':     t_dat_e2e,
        'speedup':          t_base_e2e / t_dat_e2e,
        # 分解
        't_vit_hd_ms':      t_vit_hd,
        't_vit_lr_ms':      t_vit_lr,
        't_llm_dat_pre_ms': t_llm_dat_pre,
        'llm_base_est_ms':  llm_base,
        'llm_dat_est_ms':   llm_dat,
    }


# ════════════════════════════════════════════════════════════════════════════════
# ③ Layerwise — CUDA event hook 逐层计时
# ════════════════════════════════════════════════════════════════════════════════

def layerwise_timing(model, fwd_fn, n_iters=LAYERWISE_ITERS):
    """用 CUDA event hook 测量每个 decoder layer 的平均前向时间 (ms)。"""
    layers    = model.model.language_model.layers
    N         = len(layers)
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


def run_layerwise_one(dat_model, processor, img, text, R, hr_scale,
                      dat_layer_indices, warmup=3, n_iters=LAYERWISE_ITERS):
    """对分辨率 R 做逐层计时，对比 Base(R) vs DAT(HD=R)。"""
    inp_base       = process_base(processor, img, text, R)
    inp_lr, inp_hd = process_dat(processor, img, text, R, hr_scale)

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

    # 预热
    for _ in range(warmup):
        fwd_base(); fwd_dat()
    torch.cuda.synchronize()

    t_base_lw = layerwise_timing(dat_model, fwd_base, n_iters=n_iters)
    t_dat_lw  = layerwise_timing(dat_model, fwd_dat,  n_iters=n_iters)

    N         = len(dat_model.model.language_model.layers)
    dat_set   = set(dat_layer_indices)
    std_set   = set(range(N)) - dat_set
    total_b   = sum(t_base_lw.values())
    total_d   = sum(t_dat_lw.values())

    nq_base = iids_b.shape[1]
    nq_lr   = iids_l.shape[1]

    W = 96
    print(f"\n{'═'*W}")
    print(f"Layerwise  R={R}  Base Nq={nq_base}  DAT Nq_lr={nq_lr}")
    print(f"  DAT 层: {sorted(dat_layer_indices)}  (共 {len(dat_layer_indices)} 层)")
    print(f"{'═'*W}")
    print(f"  {'层':>4}  {'类型':>5}  {'Base(ms)':>9}  {'DAT(ms)':>9}"
          f"  {'差值(ms)':>9}  {'差%':>7}  {'Base占%':>8}  {'DAT占%':>8}")
    print(f"  {'─'*84}")

    layer_rows = []
    for i in range(N):
        lt   = 'DAT' if i in dat_set else 'Std'
        tb   = t_base_lw.get(i, 0.0)
        td   = t_dat_lw.get(i, 0.0)
        diff = td - tb
        pct_diff = diff / tb * 100 if tb > 0 else 0
        pct_b = tb / total_b * 100
        pct_d = td / total_d * 100
        marker = ' ◄' if i in dat_set else ''
        print(f"  {i:>4}  {lt:>5}  {tb:>9.2f}  {td:>9.2f}"
              f"  {diff:>+9.2f}  {pct_diff:>+6.1f}%  {pct_b:>7.1f}%  {pct_d:>7.1f}%{marker}")
        layer_rows.append({'layer': i, 'type': lt,
                           't_base_ms': tb, 't_dat_ms': td,
                           'diff_ms': diff, 'diff_pct': pct_diff})

    print(f"  {'─'*84}")
    print(f"  {'合计':>4}  {'':>5}  {total_b:>9.2f}  {total_d:>9.2f}"
          f"  {total_d - total_b:>+9.2f}"
          f"  {(total_d - total_b) / total_b * 100:>+6.1f}%")

    # 分组统计
    if dat_set and std_set:
        avg_bd = sum(t_base_lw[i] for i in dat_set) / len(dat_set)
        avg_dd = sum(t_dat_lw[i]  for i in dat_set) / len(dat_set)
        avg_bs = sum(t_base_lw[i] for i in std_set) / len(std_set)
        avg_ds = sum(t_dat_lw[i]  for i in std_set) / len(std_set)
        print(f"\n  平均 DAT 层 ({len(dat_set)})  : Base={avg_bd:.2f}ms  DAT={avg_dd:.2f}ms"
              f"  diff={avg_dd-avg_bd:+.2f}ms ({(avg_dd-avg_bd)/avg_bd*100:+.1f}%)")
        print(f"  平均 Std 层 ({len(std_set)}) : Base={avg_bs:.2f}ms  DAT={avg_ds:.2f}ms"
              f"  diff={avg_ds-avg_bs:+.2f}ms ({(avg_ds-avg_bs)/avg_bs*100:+.1f}%)")
        print(f"  注: Std 层因序列长度不同 (Base Nq={nq_base} vs DAT Nq={nq_lr}) 而有速度差异")

    return {'R': R, 'nq_base': nq_base, 'nq_lr': nq_lr,
            'total_base_ms': total_b, 'total_dat_ms': total_d,
            'layers': layer_rows}


# ════════════════════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════════════════════

def run_all(dat_model, processor, images, resolutions, warmup, iters,
            hr_scale, dat_layer_indices, layerwise_rs, output_file=None):

    print(f"\n{'═'*100}")
    print("Pareto 速度测试 v2：E2E + 分解 + Layerwise")
    print(f"  Base(R)   = DAT 模型，pixel_values_hd=None，全注意力  Nq=R²/196+text")
    print(f"  DAT(HD=R) = DAT 模型，HD=R，LR=R/{hr_scale}，DAT cross-attn  Nq=LR²/196+text")
    print(f"  DAT 层索引: {sorted(dat_layer_indices)}  iters={iters}  warmup={warmup}")
    print(f"{'═'*100}")

    all_res = []
    lw_res  = []
    img = images[0]
    text = make_text(processor, img)

    for R in resolutions:
        lr = R // hr_scale
        assert R % (FACTOR * hr_scale) == 0, \
            f"R={R} 不是 {FACTOR*hr_scale} 的倍数！LR={lr} 不是 {FACTOR} 的倍数！"

        print(f"\n{'─'*100}")
        print(f"R={R}  LR={lr}  HD_tok={R*R//196}  LR_tok={lr*lr//196}")
        print(f"{'─'*100}")

        res = measure_one_resolution(dat_model, processor, img, text, R,
                                     hr_scale, warmup=warmup, iters=iters)
        all_res.append(res)

        s = res['speedup']
        print(f"\n  ┌── E2E ───────────────────────────────────────────────────")
        print(f"  │  Base(R={R})  : {res['t_base_e2e_ms']:>8.1f} ms"
              f"  Nq={res['nq_base']} (vis={res['HD_tokens']})")
        print(f"  │  DAT(HD={R}) : {res['t_dat_e2e_ms']:>8.1f} ms"
              f"  Nq_lr={res['nq_lr']} (vis_lr={res['LR_tokens']}, vis_hd={res['HD_tokens']})")
        print(f"  │  加速比       : {s:.3f}x  ({'DAT 更快' if s > 1 else 'Base 更快'}"
              f"  差值 {abs(res['t_base_e2e_ms']-res['t_dat_e2e_ms']):.1f} ms)")
        print(f"  ├── 分解 ─────────────────────────────────────────────────────")
        print(f"  │  ViT(HD=R)    : {res['t_vit_hd_ms']:>8.1f} ms  (Base ViT = DAT HD ViT，同一分辨率 R)")
        print(f"  │  ViT(LR=R/{hr_scale}) : {res['t_vit_lr_ms']:>8.1f} ms  (DAT LR ViT)")
        print(f"  │  LLM_Base(≈)  : {res['llm_base_est_ms']:>8.1f} ms  (≈ Base E2E - ViT_HD)")
        print(f"  │  LLM_DAT(≈)   : {res['llm_dat_est_ms']:>8.1f} ms  (≈ LR_ViT+LLM_DAT - ViT_LR)")
        print(f"  │  LLM_DAT+LR_ViT: {res['t_llm_dat_pre_ms']:>7.1f} ms  (直接测量：LR ViT + LLM cross-attn)")
        if res['llm_base_est_ms'] > 0 and res['llm_dat_est_ms'] > 0:
            llm_sp = res['llm_base_est_ms'] / res['llm_dat_est_ms']
            print(f"  └── LLM 加速比  : {llm_sp:.2f}x  (LLM_Base / LLM_DAT，序列缩短 {res['nq_base']/res['nq_lr']:.1f}x)")
        else:
            print(f"  └──")

        # Layerwise（仅对指定分辨率）
        if R in layerwise_rs:
            lw = run_layerwise_one(dat_model, processor, img, text, R, hr_scale,
                                   dat_layer_indices, warmup=warmup,
                                   n_iters=LAYERWISE_ITERS)
            lw_res.append(lw)

    # ── 汇总表格 ─────────────────────────────────────────────────────────────
    W = 110
    print(f"\n{'═'*W}")
    print("E2E + 分解 汇总")
    print(f"  {'R':>5}  {'LR':>5}  {'HD_tok':>7}  {'LR_tok':>7} │"
          f" {'Base E2E':>10}  {'DAT E2E':>9}  {'加速比':>7} │"
          f" {'ViT_HD':>8}  {'ViT_LR':>8}  {'LLM_Base≈':>10}  {'LLM_DAT≈':>10}")
    print(f"  {'─'*W}")
    for r in all_res:
        print(f"  {r['R']:>5}  {r['LR']:>5}  {r['HD_tokens']:>7}  {r['LR_tokens']:>7} │"
              f"  {r['t_base_e2e_ms']:>9.1f}  {r['t_dat_e2e_ms']:>8.1f}  {r['speedup']:>6.3f}x │"
              f"  {r['t_vit_hd_ms']:>7.1f}  {r['t_vit_lr_ms']:>7.1f}"
              f"  {r['llm_base_est_ms']:>9.1f}  {r['llm_dat_est_ms']:>9.1f}")

    if output_file:
        out = {'e2e_breakdown': all_res, 'layerwise': lw_res}
        with open(output_file, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  结果已保存 → {output_file}")

    return all_res, lw_res


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pareto 速度测试 v2：E2E + 分解 + Layerwise"
    )
    parser.add_argument("--base-model",  type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dat-ckpt",    type=str, default=DEFAULT_DAT_CKPT)
    parser.add_argument("--vstar-dir",   type=str, default=DEFAULT_VSTAR_DIR)
    parser.add_argument("--vstar-jsonl", type=str, default=DEFAULT_VSTAR_JSONL)
    parser.add_argument("--synthetic",   action="store_true",
                        help="使用合成图像（不需要数据集）")
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=DEFAULT_RESOLUTIONS,
                        help="待测分辨率列表（必须是 84 的倍数）")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--warmup",      type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters",       type=int, default=DEFAULT_ITERS)
    parser.add_argument("--no-layerwise", action="store_true",
                        help="跳过逐层测速（更快）")
    parser.add_argument("--layerwise-only-r", type=int, nargs="+", default=None,
                        help="只对指定分辨率做 layerwise（默认对所有分辨率做）")
    parser.add_argument("--output",      type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        Qwen2_5_VLDATForConditionalGeneration, _FA_BACKEND,
    )

    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.version.cuda}")
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"FA 后端  : {_FA_BACKEND}")

    processor = AutoProcessor.from_pretrained(args.dat_ckpt or args.base_model)

    print(f"\n加载 DAT 模型 → {args.dat_ckpt}")
    dat_model = Qwen2_5_VLDATForConditionalGeneration.from_pretrained(
        args.dat_ckpt, torch_dtype=DTYPE, device_map={"": 0},
    ).eval()

    dat_ea     = dat_model.config.dat_extra_args
    hr_scale   = dat_ea.get('hr_scale', HR_SCALE)
    layers_str = dat_ea.get('layers', '')
    dat_layer_indices = [i for i, c in enumerate(layers_str) if c == 'D']

    print(f"  hr_scale={hr_scale}, DAT 层={dat_layer_indices}")
    print(f"  显存: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    # 验证分辨率
    unit = FACTOR * hr_scale
    for R in args.resolutions:
        assert R % unit == 0, f"R={R} 不是 {unit}(=28×{hr_scale}) 的倍数"

    # 加载图像
    if args.synthetic or not os.path.exists(args.vstar_jsonl):
        max_R = max(args.resolutions)
        print(f"\n使用合成图像 {max_R}×{max_R}")
        images = [Image.new("RGB", (max_R, max_R), color=(100, 149, 237))]
    else:
        print(f"\n加载 vstar 图像（前 {args.max_samples} 张）…")
        images = []
        with open(args.vstar_jsonl) as f:
            for line in f:
                if len(images) >= args.max_samples:
                    break
                item = json.loads(line.strip())
                ip = os.path.join(args.vstar_dir, item['image'])
                if os.path.exists(ip):
                    try:
                        images.append(Image.open(ip).convert("RGB"))
                    except Exception:
                        pass
        print(f"  加载 {len(images)} 张图像")
        if not images:
            max_R = max(args.resolutions)
            print(f"  未找到图像，改用合成图像 {max_R}×{max_R}")
            images = [Image.new("RGB", (max_R, max_R), color=(100, 149, 237))]

    # 确定 layerwise 分辨率
    if args.no_layerwise:
        layerwise_rs = set()
    elif args.layerwise_only_r:
        layerwise_rs = set(args.layerwise_only_r)
    else:
        layerwise_rs = set(args.resolutions)

    out = args.output or f"pareto_speed_{time.strftime('%Y%m%d_%H%M%S')}.json"
    run_all(dat_model, processor, images,
            resolutions=args.resolutions,
            warmup=args.warmup,
            iters=args.iters,
            hr_scale=hr_scale,
            dat_layer_indices=dat_layer_indices,
            layerwise_rs=layerwise_rs,
            output_file=out)

    print("\n✓ 测速完成")


if __name__ == "__main__":
    main()
