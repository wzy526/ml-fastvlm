"""
test_dat_full_speed.py  (v2)

Qwen2.5-VL-DAT 端到端测速脚本（与训练配置对齐）。

三项基准：
  A) 单层 Attention 速度
       从 DAT 模型取出 DAT attention 层，用真实 vstar 序列长度测速。
       DAT 路径（含 HD cross-attention）vs Baseline（无 HD，退化为标准 causal attention）。
       flash_attn vs manual SDPA 对比。

  B) 全模型 Prefill 延迟（DAT vs Baseline）
       DAT 模型传入 pixel_values_hd → HD 视觉编码 + 两路 attention + LSE merge。
       同一模型不传 pixel_values_hd → 退化为标准 attention（Baseline）。
       拆分：HD-enc（ViT HD 路）/ LLM-only（预计算 HD features）/ 端到端 / Baseline。

  C) 逐层耗时分布（CUDA event hook）
       对 DAT 模型（with HD）和 Baseline（no HD），用 CUDA event 测量每个 decoder layer
       的前向时间，显示 DAT 层（6 个）与标准层（30 个）的逐层开销。

配置与训练脚本对齐：
  来源：train_dat_qwen2_5vl_z3_1d5l_s12_g8_i128_inten_gate_hd_prec_1M.sh
  DAT_LAYERS = "DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"  (1D5L，DAT 层 0,6,12,18,24,30)
  grid_size=12, off_grps=8, inter_size=128, hr_scale=3

v1 → v2 修复/改进：
  1. DAT_EXTRA_ARGS 与训练脚本对齐（off_grps=8, grid_size=12, layers=1D5L）。
  2. Part A 的 hd_feat 尺寸修正：使用 _generate_hd_features 的实际输出尺寸
     [thw_hd[1]//MERGE_SIZE, thw_hd[2]//MERGE_SIZE]，而非 lr_h_tok*HR_SCALE（偏大 2x）。
  3. probe_sizes 元组增加 hd_h_feat / hd_w_feat 字段。
  4. 合成 case 的 ans_end 不再越界（clamp 到 Nq-1）。
  5. 新增 Part C 逐层耗时（CUDA event hook）。
  6. 全模型基准增加 Baseline 列（pixel_values_hd=None）对比。

hd_h / lr_h 计算约定（与训练 Qwen2VLCoupledDATDataset._get_item 完全一致）：
  thw = inputs_hd["image_grid_thw"][0]   # (t, h_raw_patches, w_raw_patches)
  hd_h = thw[1] * FACTOR                 # FACTOR = 28 = PATCH_SIZE * MERGE_SIZE
  lr_h = max(FACTOR, (hd_h // HR_SCALE // FACTOR) * FACTOR)
  注：hd_h 为"等效像素高"（≈ 2× 实际 HD 像素高），但与训练耦合逻辑一致。
  _generate_hd_features 实际 HD feature 高度 = thw[1] // MERGE_SIZE（始终为 hd_h//56）。
"""

import sys, os, time, json, contextlib
from collections import defaultdict

import torch
from PIL import Image

sys.path.insert(0, '/root/ml-fastvlm')

# ─── 路径 ──────────────────────────────────────────────────────────────────────
BASE_MODEL  = '/data/base_models/Qwen2.5-VL-3B-Instruct'
VSTAR_DIR   = '/data/sft_data/vstar_bench'
VSTAR_JSONL = os.path.join(VSTAR_DIR, 'test_questions.jsonl')

# ─── 训练配置对齐 ─────────────────────────────────────────────────────────────
# 来源：train_dat_qwen2_5vl_z3_1d5l_s12_g8_i128_inten_gate_hd_prec_1M.sh
# 36 层 Qwen2.5-VL-3B，1D5L 模式：DAT 层索引 0,6,12,18,24,30（共 6 层）
DAT_LAYERS = "DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"
DAT_LAYER_INDICES = [i for i, c in enumerate(DAT_LAYERS) if c == 'D']  # [0,6,12,18,24,30]

DAT_EXTRA_ARGS = {
    'grid_size':            12,   # dat_grid_size 12
    'off_ksize':             3,
    'off_grps':              8,   # dat_off_grps  8
    'inter_size':          128,   # dat_inter_size 128
    'hr_scale':              3,   # dat_hr_scale  3
    'hd_proj':            True,   # dat_hd_proj   True
    'use_intention_branch': True,
    'intention_as_gate':   True,
    'layers':           DAT_LAYERS,
}

PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR     = PATCH_SIZE * MERGE_SIZE   # 28 = 像素/token 边长
HR_SCALE   = DAT_EXTRA_ARGS['hr_scale']

MAX_SAMPLES = 20
WARMUP      = 3
ITERS       = 8
DTYPE       = torch.bfloat16
DEVICE      = 'cuda'


# ════════════════════════════════════════════════════════════════════════════════
# 图像预处理（与训练 Qwen2VLCoupledDATDataset._get_item 完全对齐）
# ════════════════════════════════════════════════════════════════════════════════
def process_sample(processor, img_path, question, device=DEVICE):
    """
    耦合 LR/HD 双路处理，与训练流程完全一致。

    返回：
      inputs_lr      : LR 路完整输入（input_ids, pixel_values, image_grid_thw, ...）
      inputs_hd      : HD 路图像输入（pixel_values, image_grid_thw）
      (lr_h_tok,
       lr_w_tok)     : LR merge 后每维 token 数
      (hd_h_feat,
       hd_w_feat)    : _generate_hd_features 实际输出的 HD feature 每维 token 数
                       = (thw_hd[1]//MERGE_SIZE, thw_hd[2]//MERGE_SIZE)
    """
    img = Image.open(img_path).convert('RGB')

    # Step 1: HD — processor 默认分辨率
    inputs_hd = processor(
        images=[img], text=["<|im_start|>"],
        return_tensors="pt", padding=False,
    )

    # Step 2: 由 HD grid 推导 LR 尺寸（与训练代码第 598-603 行完全一致）
    thw    = inputs_hd["image_grid_thw"][0]          # (t, h_raw_patches, w_raw_patches)
    hd_h   = thw[1].item() * FACTOR                  # 训练约定的"等效 HD 高度"
    hd_w   = thw[2].item() * FACTOR
    lr_h   = max(FACTOR, (hd_h // HR_SCALE // FACTOR) * FACTOR)
    lr_w   = max(FACTOR, (hd_w // HR_SCALE // FACTOR) * FACTOR)
    lr_tot = lr_h * lr_w

    lr_h_tok = lr_h // FACTOR
    lr_w_tok = lr_w // FACTOR

    # HD feature 实际尺寸（= _generate_hd_features 的输出维度）
    hd_h_feat = thw[1].item() // MERGE_SIZE          # h_raw_patches // 2
    hd_w_feat = thw[2].item() // MERGE_SIZE

    # Step 3: LR — 强制分辨率
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": question + "\nAnswer with the option's letter directly."},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs_lr = processor(
        images=[img], text=[text],
        return_tensors="pt", padding=False,
        min_pixels=lr_tot, max_pixels=lr_tot,
    )

    return (
        {k: v.to(device) for k, v in inputs_lr.items()},
        {k: v.to(device) for k, v in inputs_hd.items()},
        (lr_h_tok, lr_w_tok),
        (hd_h_feat, hd_w_feat),
    )


# ════════════════════════════════════════════════════════════════════════════════
# 计时工具
# ════════════════════════════════════════════════════════════════════════════════
def benchmark_fn(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000   # ms/iter


import llava.model.language_model.modeling_qwen2_5vl_dat as _dat_mod


@contextlib.contextmanager
def manual_sdpa_mode():
    """临时关闭 flash_attn，切换为 manual SDPA（monkey-patch 模块全局变量）。"""
    orig = _dat_mod._FLASH_ATTN_AVAILABLE
    _dat_mod._FLASH_ATTN_AVAILABLE = False
    try:
        yield
    finally:
        _dat_mod._FLASH_ATTN_AVAILABLE = orig


# ════════════════════════════════════════════════════════════════════════════════
# CUDA Event 逐层 Hook
# ════════════════════════════════════════════════════════════════════════════════
def layerwise_timing(model, fwd_fn, n_iters=5):
    """
    用 CUDA event hook 测量每个 decoder layer 的平均前向时间（ms）。

    Args:
        model   : DAT 模型
        fwd_fn  : callable，内部调用 model.forward（包含 torch.no_grad）
        n_iters : 计时迭代次数

    Returns:
        dict[int → float]: layer_idx → 平均耗时 ms
    """
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


# ════════════════════════════════════════════════════════════════════════════════
# Part A：单层 Attention 基准测试
# ════════════════════════════════════════════════════════════════════════════════
def run_layer_benchmark(dat_model, sample_sizes, device=DEVICE, dtype=DTYPE):
    """
    从 dat_model 取第一个 DAT attention 层，用合成 hidden_states 测速。

    sample_sizes: list of
      (label, B, Nq, lr_h_tok, lr_w_tok, hd_h_feat, hd_w_feat, ans_len, grid_size)

    hd_h_feat / hd_w_feat：_generate_hd_features 的实际输出维度
      = (thw_hd[1] // MERGE_SIZE, thw_hd[2] // MERGE_SIZE)
    注意：这与 lr_h_tok * HR_SCALE 不同（约差 2x）！
    """
    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        Qwen2_5_VLAttentionDAT, _FLASH_ATTN_AVAILABLE,
    )

    text_cfg = dat_model.config.text_config
    head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
    C        = text_cfg.hidden_size

    dat_layer = next(
        (l.self_attn for l in dat_model.model.language_model.layers
         if isinstance(getattr(l, 'self_attn', None), Qwen2_5_VLAttentionDAT)),
        None,
    )
    assert dat_layer is not None, "No DAT attention layer found"
    dat_layer.eval()

    fa_tag = 'YES' if _FLASH_ATTN_AVAILABLE else 'NO'
    print(f"\n{'═'*104}")
    print(f"Part A — 单层 Attention 速度  (flash_attn={fa_tag},"
          f" DAT 层索引={DAT_LAYER_INDICES}, grid_size={DAT_EXTRA_ARGS['grid_size']},"
          f" off_grps={DAT_EXTRA_ARGS['off_grps']})")
    print(f"  DAT=两路attn+LSE merge  |  Baseline=标准因果attn（无HD）")
    print(f"{'─'*104}")
    print(f"  {'Case':<34} {'Nq':>5}  {'LR':>7}  {'HD feat':>9}"
          f"  {'DAT-fa':>8}  {'DAT-man':>8}  {'fa加速':>7}"
          f"  {'Baseline':>9}  {'DAT开销':>10}")
    print(f"{'─'*104}")

    for row in sample_sizes:
        label, B, Nq, lr_h_tok, lr_w_tok, hd_h_feat, hd_w_feat, ans_len, grid_size = row

        lr_len   = lr_h_tok * lr_w_tok
        lr_start = 1
        lr_end   = lr_start + lr_len

        # 确保 ans_end 不越界
        act_ans = min(ans_len, max(1, Nq - lr_end - 6))
        if act_ans <= 0:
            print(f"  {label:<34}  [跳过，Nq={Nq} 太短，需 > {lr_end + 7}]")
            continue

        hidden    = torch.randn(B, Nq, C,           device=device, dtype=dtype)
        cos       = torch.ones (3, B, Nq, head_dim, device=device, dtype=dtype)
        sin       = torch.zeros(3, B, Nq, head_dim, device=device, dtype=dtype)
        pos_emb   = (cos, sin)
        mrope_pos = (torch.arange(Nq, device=device, dtype=torch.long)
                     .unsqueeze(0).unsqueeze(0).expand(3, B, -1))

        # HD feature：使用 _generate_hd_features 的实际输出尺寸
        hd_feat = torch.randn(hd_h_feat, hd_w_feat, C, device=device, dtype=dtype)

        def irl():
            # image_range_list: 每次生成新对象，避免多次调用间的引用共享问题
            return [
                [(lr_start, lr_end, lr_h_tok, lr_w_tok),
                 [lr_end + 5, lr_end + 5 + act_ans, lr_end + 1]]
            ]

        def run_dat():
            with torch.no_grad():
                dat_layer(
                    hidden,
                    position_embeddings=pos_emb,
                    image_hd_features=[hd_feat],
                    image_range_list=irl(),
                    mrope_position_ids=mrope_pos,
                )

        def run_baseline():
            with torch.no_grad():
                dat_layer(
                    hidden,
                    position_embeddings=pos_emb,
                    image_hd_features=None,
                    image_range_list=None,
                    mrope_position_ids=mrope_pos,
                )

        t_dat_fa  = benchmark_fn(run_dat)
        t_baseline = benchmark_fn(run_baseline)
        with manual_sdpa_mode():
            t_dat_man = benchmark_fn(run_dat)

        fa_spd   = t_dat_man / t_dat_fa
        dat_ovhd = (t_dat_fa - t_baseline) / t_baseline * 100
        hd_toks  = hd_h_feat * hd_w_feat

        print(f"  {label:<34} {Nq:>5}  {lr_h_tok}×{lr_w_tok:<4}"
              f"  {hd_h_feat}×{hd_w_feat:<5}"
              f"  {t_dat_fa:>8.2f}  {t_dat_man:>8.2f}  {fa_spd:>6.2f}x"
              f"  {t_baseline:>9.2f}  {dat_ovhd:>+9.1f}%")

    print(f"{'─'*104}")


# ════════════════════════════════════════════════════════════════════════════════
# Part B：全模型 Prefill 延迟（DAT vs Baseline）
# ════════════════════════════════════════════════════════════════════════════════
def run_full_model_benchmark(dat_model, processor, vstar_samples,
                             device=DEVICE, dtype=DTYPE):
    """
    对比（同一 DAT 模型）：
      · DAT prefill    : pixel_values_hd → HD ViT 编码 + 两路 attention + LSE merge
      · Baseline        : pixel_values_hd=None → 退化为标准因果 attention（等价于原版模型）

    额外拆分：
      · HD 视觉编码（_generate_hd_features，含 ViT forward）
      · LLM-only DAT（预计算 HD features，仅测 LLM forward）
      · LLM-only flash vs manual SDPA 加速比
    """
    print(f"\n{'═'*104}")
    print("Part B — 全模型 Prefill 延迟（DAT vs Baseline，真实 vstar 图像）")
    print(f"  Baseline = 同模型，pixel_values_hd=None（DAT 层退化为标准 attention）")
    print(f"{'═'*104}")
    print(f"  {'#':>3}  {'Nq':>4}  {'LR':>6}  {'HD feat':>8}"
          f"  {'DAT':>8}  {'HD-enc':>8}  {'LLM-fa':>8}  {'LLM-man':>9}"
          f"  {'fa加速':>7}  {'Baseline':>9}  {'LLM净开销':>10}")
    print(f"  {'─'*98}")

    results = defaultdict(list)
    seq_lens = []

    # ── 全局 warmup ──────────────────────────────────────────────────────────
    print("  [全局 warmup] 预热全模型路径…", end="", flush=True)
    warmup_done = False
    for ws in vstar_samples:
        wimg = os.path.join(VSTAR_DIR, ws['image'])
        if not os.path.exists(wimg):
            continue
        try:
            wi_lr, wi_hd, _, _ = process_sample(processor, wimg, ws['text'], device=device)
            pv_hd = wi_hd['pixel_values'].to(dtype); thw_hd = wi_hd['image_grid_thw']
            pv_lr = wi_lr['pixel_values'].to(dtype); thw_lr = wi_lr['image_grid_thw']
            iids  = wi_lr['input_ids'];              amask  = wi_lr.get('attention_mask')
            for _ in range(3):
                with torch.no_grad():
                    dat_model(input_ids=iids, attention_mask=amask,
                              pixel_values=pv_lr, image_grid_thw=thw_lr,
                              pixel_values_hd=pv_hd, image_grid_thw_hd=thw_hd,
                              use_cache=False)
                    dat_model(input_ids=iids, attention_mask=amask,
                              pixel_values=pv_lr, image_grid_thw=thw_lr,
                              pixel_values_hd=None, image_grid_thw_hd=None,
                              use_cache=False)
            torch.cuda.synchronize()
            warmup_done = True
            break
        except Exception:
            continue
    print(" done" if warmup_done else " skipped")

    for i, sample in enumerate(vstar_samples):
        img_path = os.path.join(VSTAR_DIR, sample['image'])
        if not os.path.exists(img_path):
            continue
        try:
            inputs_lr, inputs_hd, lr_hw, hd_hw = process_sample(
                processor, img_path, sample['text'], device=device)
        except Exception as e:
            print(f"  [skip {i}] {e}")
            continue

        Nq = inputs_lr['input_ids'].shape[1]
        seq_lens.append(Nq)

        pv_hd  = inputs_hd['pixel_values'].to(dtype)
        thw_hd = inputs_hd['image_grid_thw']
        pv_lr  = inputs_lr['pixel_values'].to(dtype)
        thw_lr = inputs_lr['image_grid_thw']
        iids   = inputs_lr['input_ids']
        amask  = inputs_lr.get('attention_mask')

        # 预计算 HD features（排除 ViT 编码时间来单独测 LLM forward）
        with torch.no_grad():
            hd_feats = dat_model._generate_hd_features(pv_hd, thw_hd)

        def fwd_hd_enc():
            with torch.no_grad():
                dat_model._generate_hd_features(pv_hd, thw_hd)

        def fwd_llm_dat():
            with torch.no_grad():
                dat_model(input_ids=iids, attention_mask=amask,
                          pixel_values=pv_lr, image_grid_thw=thw_lr,
                          image_hd_features=hd_feats, use_cache=False)

        def fwd_dat():
            with torch.no_grad():
                dat_model(input_ids=iids, attention_mask=amask,
                          pixel_values=pv_lr, image_grid_thw=thw_lr,
                          pixel_values_hd=pv_hd, image_grid_thw_hd=thw_hd,
                          use_cache=False)

        def fwd_baseline():
            with torch.no_grad():
                dat_model(input_ids=iids, attention_mask=amask,
                          pixel_values=pv_lr, image_grid_thw=thw_lr,
                          pixel_values_hd=None, image_grid_thw_hd=None,
                          use_cache=False)

        t_hd_enc  = benchmark_fn(fwd_hd_enc,   warmup=2, iters=5)
        t_llm_fa  = benchmark_fn(fwd_llm_dat,  warmup=2, iters=5)
        t_dat     = benchmark_fn(fwd_dat,      warmup=2, iters=5)
        t_base    = benchmark_fn(fwd_baseline, warmup=2, iters=5)
        with manual_sdpa_mode():
            t_llm_man = benchmark_fn(fwd_llm_dat, warmup=2, iters=5)

        fa_spd  = t_llm_man / t_llm_fa
        net_llm = t_llm_fa - t_base   # LLM-only DAT 相对 Baseline 的净开销

        results['dat'].append(t_dat)
        results['base'].append(t_base)
        results['hd_enc'].append(t_hd_enc)
        results['llm_fa'].append(t_llm_fa)
        results['llm_man'].append(t_llm_man)

        hd_h_feat, hd_w_feat = hd_hw
        print(f"  [{i:02d}] {Nq:4d}  {lr_hw[0]}×{lr_hw[1]:<3d}  "
              f"{hd_h_feat}×{hd_w_feat:<4d}"
              f"  {t_dat:>8.1f}  {t_hd_enc:>8.1f}  {t_llm_fa:>8.1f}  {t_llm_man:>9.1f}"
              f"  {fa_spd:>6.2f}x  {t_base:>9.1f}  {net_llm:>+9.1f}")

    if not results['dat']:
        print("  没有成功处理的样本，跳过 Part B")
        return

    def avg(k): return sum(results[k]) / len(results[k])
    aN        = len(results['dat'])
    a_dat     = avg('dat'); a_base = avg('base')
    a_hd      = avg('hd_enc')
    a_llm_fa  = avg('llm_fa'); a_llm_man = avg('llm_man')
    a_nq      = sum(seq_lens) / aN
    a_fa_spd  = a_llm_man / a_llm_fa

    print(f"\n{'─'*104}")
    print(f"  样本数: {aN}  平均 Nq={a_nq:.0f}")
    print(f"  Baseline prefill         : {a_base:.1f} ms   （标准 causal attn，无 HD）")
    print(f"  HD 视觉编码 (ViT HD路)   : {a_hd:.1f} ms   （_generate_hd_features）")
    print(f"  LLM-only DAT flash       : {a_llm_fa:.1f} ms   （预计算 HD features）")
    print(f"  LLM-only DAT manual SDPA : {a_llm_man:.1f} ms")
    print(f"  flash vs manual 加速比   : {a_fa_spd:.2f}x")
    print(f"  DAT 端到端               : {a_dat:.1f} ms")
    print(f"  LLM 净 DAT 开销 (flash)  : {a_llm_fa - a_base:+.1f} ms"
          f"  ({(a_llm_fa - a_base) / a_base * 100:+.1f}%)")
    print(f"{'─'*104}")


# ════════════════════════════════════════════════════════════════════════════════
# Part C：逐层耗时分布（CUDA event hook）
# ════════════════════════════════════════════════════════════════════════════════
def run_layerwise_benchmark(dat_model, processor, vstar_samples,
                            n_iters=5, device=DEVICE, dtype=DTYPE):
    """
    对 DAT（with HD）和 Baseline（no HD）分别用 CUDA event 测量每个 decoder layer
    的前向时间，输出逐层对比表。
    """
    print(f"\n{'═'*104}")
    print(f"Part C — 逐层耗时分布（CUDA event hook，{n_iters} iters）")
    print(f"  DAT 层索引: {DAT_LAYER_INDICES}  (共 {len(DAT_LAYER_INDICES)} 层)")
    print(f"{'═'*104}")

    # 找第一个有效样本
    inputs_lr = inputs_hd = None
    for sample in vstar_samples:
        img_path = os.path.join(VSTAR_DIR, sample['image'])
        if not os.path.exists(img_path):
            continue
        try:
            inputs_lr, inputs_hd, _, _ = process_sample(
                processor, img_path, sample['text'], device=device)
            break
        except Exception:
            continue

    if inputs_lr is None:
        print("  没有可用样本，跳过 Part C")
        return

    pv_hd  = inputs_hd['pixel_values'].to(dtype)
    thw_hd = inputs_hd['image_grid_thw']
    pv_lr  = inputs_lr['pixel_values'].to(dtype)
    thw_lr = inputs_lr['image_grid_thw']
    iids   = inputs_lr['input_ids']
    amask  = inputs_lr.get('attention_mask')
    Nq     = iids.shape[1]

    with torch.no_grad():
        hd_feats = dat_model._generate_hd_features(pv_hd, thw_hd)

    def fwd_dat():
        with torch.no_grad():
            dat_model(input_ids=iids, attention_mask=amask,
                      pixel_values=pv_lr, image_grid_thw=thw_lr,
                      image_hd_features=hd_feats, use_cache=False)

    def fwd_baseline():
        with torch.no_grad():
            dat_model(input_ids=iids, attention_mask=amask,
                      pixel_values=pv_lr, image_grid_thw=thw_lr,
                      pixel_values_hd=None, image_grid_thw_hd=None,
                      use_cache=False)

    # warmup
    with torch.no_grad():
        for _ in range(3):
            fwd_dat()
            fwd_baseline()
    torch.cuda.synchronize()

    print(f"  测量中（Nq={Nq}）…", end="", flush=True)
    t_dat  = layerwise_timing(dat_model, fwd_dat,      n_iters=n_iters)
    t_base = layerwise_timing(dat_model, fwd_baseline, n_iters=n_iters)
    print(" done")

    N          = len(dat_model.model.language_model.layers)
    total_dat  = sum(t_dat.values())
    total_base = sum(t_base.values())
    dat_set    = set(DAT_LAYER_INDICES)
    std_set    = set(range(N)) - dat_set

    print(f"\n  {'层':>4}  {'类型':>5}  {'DAT(ms)':>9}  {'Base(ms)':>9}"
          f"  {'开销(ms)':>9}  {'开销%':>7}  {'占总时%':>7}")
    print(f"  {'─'*66}")

    for i in range(N):
        layer_type = 'DAT' if i in dat_set else 'Std'
        td = t_dat.get(i, 0.0); tb = t_base.get(i, 0.0)
        diff = td - tb
        pct_ovhd = diff / tb * 100 if tb > 0 else 0
        pct_tot  = td / total_dat * 100
        marker   = ' ◄' if i in dat_set else ''
        print(f"  {i:>4}  {layer_type:>5}  {td:>9.2f}  {tb:>9.2f}"
              f"  {diff:>+9.2f}  {pct_ovhd:>+6.1f}%  {pct_tot:>6.1f}%{marker}")

    print(f"  {'─'*66}")
    print(f"  {'合计':>4}  {'':>5}  {total_dat:>9.2f}  {total_base:>9.2f}"
          f"  {total_dat - total_base:>+9.2f}"
          f"  {(total_dat - total_base) / total_base * 100:>+6.1f}%")

    # 分组统计
    avg_dat_d  = sum(t_dat[i]  for i in dat_set) / len(dat_set)
    avg_base_d = sum(t_base[i] for i in dat_set) / len(dat_set)
    avg_dat_s  = sum(t_dat[i]  for i in std_set) / len(std_set)
    avg_base_s = sum(t_base[i] for i in std_set) / len(std_set)

    print(f"\n  平均每 DAT 层（{len(dat_set)} 层）  :"
          f" {avg_dat_d:.2f} ms  vs  Baseline {avg_base_d:.2f} ms"
          f"  (开销 {avg_dat_d - avg_base_d:+.2f} ms /"
          f" {(avg_dat_d - avg_base_d) / avg_base_d * 100:+.1f}%)")
    print(f"  平均每标准层（{len(std_set)} 层） :"
          f" {avg_dat_s:.2f} ms  vs  Baseline {avg_base_s:.2f} ms"
          f"  (开销 {avg_dat_s - avg_base_s:+.2f} ms /"
          f" {(avg_dat_s - avg_base_s) / avg_base_s * 100:+.1f}%)")
    print(f"{'─'*104}")


# ════════════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════════════
def main():
    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        convert_qwen2_5vl_to_dat, _FLASH_ATTN_AVAILABLE,
    )

    print(f"\nPyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.version.cuda}")
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"flash_attn: {'YES' if _FLASH_ATTN_AVAILABLE else 'NO'}")
    print(f"\n训练配置：")
    print(f"  DAT_LAYERS   = {DAT_LAYERS}")
    print(f"  DAT 层索引   = {DAT_LAYER_INDICES}  (共 {len(DAT_LAYER_INDICES)} 层)")
    print(f"  grid_size={DAT_EXTRA_ARGS['grid_size']},"
          f" off_grps={DAT_EXTRA_ARGS['off_grps']},"
          f" inter_size={DAT_EXTRA_ARGS['inter_size']},"
          f" hr_scale={HR_SCALE}")

    # ── 加载 vstar 样本 ───────────────────────────────────────────────────────
    print(f"\n加载 vstar 样本（前 {MAX_SAMPLES} 个）…")
    vstar_samples = []
    with open(VSTAR_JSONL) as f:
        for line in f:
            vstar_samples.append(json.loads(line.strip()))
            if len(vstar_samples) >= MAX_SAMPLES:
                break
    print(f"  共 {len(vstar_samples)} 个样本")

    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    # ── 加载 DAT 模型（与训练配置对齐：1D5L，6 个 DAT 层）──────────────────────
    print(f"\n加载并转换模型 → Qwen2.5-VL-3B-DAT"
          f"（{len(DAT_LAYER_INDICES)} 个 DAT 层：{DAT_LAYER_INDICES}）…")
    dat_model = convert_qwen2_5vl_to_dat(
        BASE_MODEL,
        dat_extra_args=DAT_EXTRA_ARGS,
        torch_dtype=DTYPE,
    ).to(device=DEVICE).eval()
    print(f"  模型加载完毕，显存: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    # ════════════════════════════════════════════════════════════════════════
    # Part A：用真实 vstar 图像推算序列/HD 特征尺寸，做单层 attention 测速
    # ════════════════════════════════════════════════════════════════════════
    print("\n推算代表性序列长度（取前 8 张 vstar 图像）…")
    probe_sizes = []
    for sample in vstar_samples[:8]:
        img_path = os.path.join(VSTAR_DIR, sample['image'])
        if not os.path.exists(img_path):
            continue
        try:
            inputs_lr, _, lr_hw, hd_hw = process_sample(
                processor, img_path, sample['text'], device=DEVICE)
            Nq = inputs_lr['input_ids'].shape[1]
            lr_h_tok, lr_w_tok   = lr_hw
            hd_h_feat, hd_w_feat = hd_hw
            lr_len  = lr_h_tok * lr_w_tok
            ans_len = max(1, Nq - lr_len - 20)
            probe_sizes.append((
                f"vstar #{len(probe_sizes):02d} ({sample['category'][:12]})",
                1, Nq, lr_h_tok, lr_w_tok, hd_h_feat, hd_w_feat,
                ans_len, DAT_EXTRA_ARGS['grid_size'],
            ))
            print(f"  {sample['image']}: Nq={Nq},"
                  f" lr={lr_h_tok}×{lr_w_tok} ({lr_len} tok),"
                  f" hd_feat={hd_h_feat}×{hd_w_feat},"
                  f" ans≈{ans_len}")
        except Exception as e:
            print(f"  [skip] {e}")

    # 合成 case：覆盖中等/长序列（Nq 设置保证 ans_end 不越界）
    # lr=(22×23)=506 tok, hd_feat≈ lr_h_tok*HR_SCALE/2 ≈ 33×34
    probe_sizes += [
        # label              B   Nq   lr_h lr_w  hd_h hd_w  ans  gs
        ("合成-vstar典型",  1, 1800,  41,  41,  62,  62,  200, 12),
        ("合成-中等序列",   1,  762,  22,  23,  33,  34,  200, 12),  # ans_end=1+506+5+200=712≤762
        ("合成-长序列",     1, 1600,  22,  23,  33,  34, 1000, 12),
    ]

    run_layer_benchmark(dat_model, probe_sizes, device=DEVICE, dtype=DTYPE)

    # ════════════════════════════════════════════════════════════════════════
    # Part B：全模型 TTFT（DAT vs Baseline）
    # ════════════════════════════════════════════════════════════════════════
    run_full_model_benchmark(dat_model, processor, vstar_samples[:MAX_SAMPLES],
                             device=DEVICE, dtype=DTYPE)

    # ════════════════════════════════════════════════════════════════════════
    # Part C：逐层耗时分布
    # ════════════════════════════════════════════════════════════════════════
    run_layerwise_benchmark(dat_model, processor, vstar_samples[:5],
                            n_iters=5, device=DEVICE, dtype=DTYPE)

    # ── 峰值显存（第一个有效 vstar 样本）──────────────────────────────────────
    print(f"\n{'═'*72}")
    print("显存峰值（第一个可用 vstar 样本）")
    print(f"{'═'*72}")
    for sample in vstar_samples:
        img_path = os.path.join(VSTAR_DIR, sample['image'])
        if not os.path.exists(img_path):
            continue
        try:
            inputs_lr, inputs_hd, _, _ = process_sample(
                processor, img_path, sample['text'], device=DEVICE)
            pv_hd  = inputs_hd['pixel_values'].to(DTYPE)
            thw_hd = inputs_hd['image_grid_thw']
            pv_lr  = inputs_lr['pixel_values'].to(DTYPE)
            thw_lr = inputs_lr['image_grid_thw']
            iids   = inputs_lr['input_ids']
            amask  = inputs_lr.get('attention_mask')

            def measure_peak(fn):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                fn()
                torch.cuda.synchronize()
                return torch.cuda.max_memory_allocated() / 1024**2

            mem_dat = measure_peak(lambda: dat_model(
                input_ids=iids, attention_mask=amask,
                pixel_values=pv_lr, image_grid_thw=thw_lr,
                pixel_values_hd=pv_hd, image_grid_thw_hd=thw_hd,
                use_cache=False,
            ))
            mem_base = measure_peak(lambda: dat_model(
                input_ids=iids, attention_mask=amask,
                pixel_values=pv_lr, image_grid_thw=thw_lr,
                pixel_values_hd=None, image_grid_thw_hd=None,
                use_cache=False,
            ))
            print(f"  Nq={iids.shape[1]}"
                  f"  Peak DAT: {mem_dat:.0f} MB  |  Baseline: {mem_base:.0f} MB"
                  f"  |  额外: {mem_dat - mem_base:+.0f} MB")
            break
        except Exception as e:
            print(f"  [skip] {e}")
            continue

    print("\n✓ 测速完成")


if __name__ == '__main__':
    main()
