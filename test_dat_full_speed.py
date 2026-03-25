"""
test_dat_full_speed.py  (v3)

Qwen2.5-VL-DAT 端到端测速脚本（与训练配置对齐）。

三项基准：
  A) 单层 Attention 速度
       从 DAT 模型取出 DAT attention 层，用真实 vstar 序列长度测速。
       DAT 路径（含 HD cross-attention）vs LR-only（无 HD，退化为标准 causal attention）。
       flash_attn vs manual SDPA 对比。

  B) 全模型 Prefill 延迟（DAT vs Fair-HD vs LR-only）
       三路公平对比（均携带相同视觉信息量）：
         · DAT       : pixel_values_hd → HD ViT + LR ViT + LLM(Nq_lr) + HD cross-attn
         · Fair-HD   : HD 图像直接作为输入 tokens → HD ViT + LLM(Nq_hd) 纯因果 attn
                       与 DAT 信息量对等，是 naive 高分辨率基线
         · LR-only   : pixel_values_hd=None → 仅 LR ViT + LLM(Nq_lr)，无 HD 信息（参考下界）

  C) 逐层耗时分布（CUDA event hook）
       DAT(Nq_lr) vs Fair-HD(Nq_hd)：逐层对比，展示两种 HD 集成方案的每层开销差异。

配置与训练脚本对齐：
  来源：train_dat_qwen2_5vl_z3_1d5l_s12_g8_i128_inten_gate_hd_prec_1M.sh
  DAT_LAYERS = "DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL"  (1D5L，DAT 层 0,6,12,18,24,30)
  grid_size=12, off_grps=8, inter_size=128, hr_scale=3

v2 → v3 改进：
  Fair-HD Baseline：将 HD 图像以原生分辨率直接作为输入 token（而非仅 LR-only 退化），
  实现与 DAT 信息量对等的公平对比。
  process_sample 新增 Step 4（inputs_hd_full）：同一问题文本 + HD 分辨率约束，
  生成含 Nq_hd 图像 token 的完整输入，用于 Fair-HD 前向。

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
    'use_fused_vit':       False,  # 关闭：LR 和 HD 走独立 ViT 调用（省显存）；True 合并为一次调用
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
    耦合 LR/HD 双路处理，与训练流程完全一致。额外生成 Fair-HD 完整输入。

    返回（5-tuple）：
      inputs_lr      : LR 路完整输入（input_ids, pixel_values, image_grid_thw, ...）
      inputs_hd      : HD 路图像输入（pixel_values, image_grid_thw，仅含最小占位文本）
      (lr_h_tok,
       lr_w_tok)     : LR merge 后每维 token 数
      (hd_h_feat,
       hd_w_feat)    : _generate_hd_features 实际输出的 HD feature 每维 token 数
                       = (thw_hd[1]//MERGE_SIZE, thw_hd[2]//MERGE_SIZE)
      inputs_hd_full : Fair-HD 路完整输入（input_ids, pixel_values, image_grid_thw）
                       与 inputs_lr 使用相同问题文本，但图像以 HD 分辨率处理，
                       产生 Nq_hd >> Nq_lr 的较长序列，用于 naive-HD baseline。
    """
    img = Image.open(img_path).convert('RGB')

    # 预先构建完整对话文本（LR 和 HD-full 共用）
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": question + "\nAnswer with the option's letter directly."},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Step 1: HD — processor 默认分辨率（仅取 pixel_values / image_grid_thw）
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
    inputs_lr = processor(
        images=[img], text=[text],
        return_tensors="pt", padding=False,
        min_pixels=lr_tot, max_pixels=lr_tot,
    )

    # Step 4: HD-full（Fair-HD baseline）
    # 强制与 inputs_hd 相同的分辨率（thw[1]*thw[2]*14² 总像素），但包含完整问题文本。
    # 生成的 input_ids 含 thw[1]//2 * thw[2]//2 = hd_h_feat * hd_w_feat 个图像 token，
    # 序列长度 Nq_hd ≈ Nq_lr + (HD tokens - LR tokens)。
    hd_tot = thw[1].item() * thw[2].item() * PATCH_SIZE * PATCH_SIZE
    inputs_hd_full = processor(
        images=[img], text=[text],
        return_tensors="pt", padding=False,
        min_pixels=hd_tot, max_pixels=hd_tot,
    )

    return (
        {k: v.to(device) for k, v in inputs_lr.items()},
        {k: v.to(device) for k, v in inputs_hd.items()},
        (lr_h_tok, lr_w_tok),
        (hd_h_feat, hd_w_feat),
        {k: v.to(device) for k, v in inputs_hd_full.items()},
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
        Qwen2_5_VLAttentionDAT, _FA_BACKEND,
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

    print(f"\n{'═'*96}")
    print(f"Part A — 单层 Attention 速度  (backend={_FA_BACKEND},"
          f" DAT 层索引={DAT_LAYER_INDICES}, grid_size={DAT_EXTRA_ARGS['grid_size']},"
          f" off_grps={DAT_EXTRA_ARGS['off_grps']})")
    print(f"  DAT=两路attn+LSE merge  |  LR-only=标准因果attn（无HD）")
    print(f"{'─'*96}")
    print(f"  {'Case':<34} {'Nq':>5}  {'LR':>7}  {'HD feat':>9}"
          f"  {'DAT(ms)':>9}  {'LR-only(ms)':>12}  {'DAT开销':>10}")
    print(f"{'─'*96}")

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

        t_dat      = benchmark_fn(run_dat)
        t_baseline = benchmark_fn(run_baseline)

        dat_ovhd = (t_dat - t_baseline) / t_baseline * 100

        print(f"  {label:<34} {Nq:>5}  {lr_h_tok}×{lr_w_tok:<4}"
              f"  {hd_h_feat}×{hd_w_feat:<5}"
              f"  {t_dat:>9.2f}  {t_baseline:>12.2f}  {dat_ovhd:>+9.1f}%")

    print(f"{'─'*96}")


# ════════════════════════════════════════════════════════════════════════════════
# Part B：全模型 Prefill 延迟（DAT vs Baseline）
# ════════════════════════════════════════════════════════════════════════════════
def run_full_model_benchmark(dat_model, processor, vstar_samples,
                             device=DEVICE, dtype=DTYPE):
    """
    三路公平对比（均使用相同视觉信息量）：
      · DAT       : pixel_values + pixel_values_hd → LR ViT + HD ViT + LLM(Nq_lr, cross-attn)
      · Fair-HD   : pixel_values=HD, pixel_values_hd=None → HD ViT + LLM(Nq_hd, causal attn)
                    与 DAT 信息量对等，naive 高分辨率基线
      · LR-only   : pixel_values=LR, pixel_values_hd=None → LR ViT + LLM(Nq_lr)，无 HD（参考）

    额外拆分（DAT 专项）：
      · HD 视觉编码（_generate_hd_features，含 ViT forward）
      · LLM-only DAT（预计算 HD features，仅测 LLM forward）
      · flash vs manual SDPA 加速比
    """
    print(f"\n{'═'*116}")
    print("Part B — 全模型 Prefill 延迟（DAT vs Fair-HD vs LR-only，真实 vstar 图像）")
    print(f"  Fair-HD = 同模型，HD 图像作为普通输入 token（Nq 更长，标准因果 attn）")
    print(f"  LR-only = 同模型，pixel_values_hd=None（无 HD 信息，速度下界参考）")
    print(f"{'═'*116}")
    print(f"  {'#':>3}  {'Nq-LR':>5}  {'Nq-HD':>5}  {'LR':>6}  {'HD feat':>8}"
          f"  {'DAT':>8}  {'Fair-HD':>8}  {'LR-only':>8}"
          f"  {'DAT/FairHD':>10}  {'LLM开销(DAT)':>12}  {'LLM开销(FairHD)':>15}")
    print(f"  {'─'*113}")

    results  = defaultdict(list)
    nq_lrs   = []
    nq_hds   = []

    # ── 全局 warmup ──────────────────────────────────────────────────────────
    print("  [全局 warmup] 预热全模型路径…", end="", flush=True)
    warmup_done = False
    for ws in vstar_samples:
        wimg = os.path.join(VSTAR_DIR, ws['image'])
        if not os.path.exists(wimg):
            continue
        try:
            wi_lr, wi_hd, _, _, wi_hd_full = process_sample(processor, wimg, ws['text'], device=device)
            pv_hd  = wi_hd['pixel_values'].to(dtype);      thw_hd  = wi_hd['image_grid_thw']
            pv_lr  = wi_lr['pixel_values'].to(dtype);      thw_lr  = wi_lr['image_grid_thw']
            iids   = wi_lr['input_ids'];                   amask   = wi_lr.get('attention_mask')
            pv_hdf = wi_hd_full['pixel_values'].to(dtype); thw_hdf = wi_hd_full['image_grid_thw']
            iids_hdf = wi_hd_full['input_ids'];            amask_hdf = wi_hd_full.get('attention_mask')
            for _ in range(3):
                with torch.no_grad():
                    dat_model(input_ids=iids, attention_mask=amask,
                              pixel_values=pv_lr, image_grid_thw=thw_lr,
                              pixel_values_hd=pv_hd, image_grid_thw_hd=thw_hd,
                              use_cache=False)
                    dat_model(input_ids=iids_hdf, attention_mask=amask_hdf,
                              pixel_values=pv_hdf, image_grid_thw=thw_hdf,
                              pixel_values_hd=None, image_grid_thw_hd=None,
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
            inputs_lr, inputs_hd, lr_hw, hd_hw, inputs_hd_full = process_sample(
                processor, img_path, sample['text'], device=device)
        except Exception as e:
            print(f"  [skip {i}] {e}")
            continue

        Nq    = inputs_lr['input_ids'].shape[1]
        Nq_hd = inputs_hd_full['input_ids'].shape[1]
        nq_lrs.append(Nq)
        nq_hds.append(Nq_hd)

        pv_hd  = inputs_hd['pixel_values'].to(dtype)
        thw_hd = inputs_hd['image_grid_thw']
        pv_lr  = inputs_lr['pixel_values'].to(dtype)
        thw_lr = inputs_lr['image_grid_thw']
        iids   = inputs_lr['input_ids']
        amask  = inputs_lr.get('attention_mask')
        pv_hdf   = inputs_hd_full['pixel_values'].to(dtype)
        thw_hdf  = inputs_hd_full['image_grid_thw']
        iids_hdf = inputs_hd_full['input_ids']
        amask_hdf = inputs_hd_full.get('attention_mask')

        # 预计算 HD features（排除 ViT 编码时间来单独测 LLM-DAT forward）
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

        def fwd_fair_hd():
            with torch.no_grad():
                dat_model(input_ids=iids_hdf, attention_mask=amask_hdf,
                          pixel_values=pv_hdf, image_grid_thw=thw_hdf,
                          pixel_values_hd=None, image_grid_thw_hd=None,
                          use_cache=False)

        def fwd_lr_only():
            with torch.no_grad():
                dat_model(input_ids=iids, attention_mask=amask,
                          pixel_values=pv_lr, image_grid_thw=thw_lr,
                          pixel_values_hd=None, image_grid_thw_hd=None,
                          use_cache=False)

        t_hd_enc  = benchmark_fn(fwd_hd_enc,  warmup=2, iters=5)
        t_llm_dat = benchmark_fn(fwd_llm_dat, warmup=2, iters=5)
        t_dat     = benchmark_fn(fwd_dat,     warmup=2, iters=5)
        t_fair    = benchmark_fn(fwd_fair_hd, warmup=2, iters=5)
        t_lr      = benchmark_fn(fwd_lr_only, warmup=2, iters=5)

        llm_dat_ovhd  = t_llm_dat - t_lr   # DAT LLM-only 净开销 vs LR-only
        llm_fair_ovhd = t_fair - t_lr       # Fair-HD 净开销 vs LR-only
        dat_vs_fair   = t_dat / t_fair      # DAT E2E vs Fair-HD E2E

        results['dat'].append(t_dat)
        results['fair'].append(t_fair)
        results['lr'].append(t_lr)
        results['hd_enc'].append(t_hd_enc)
        results['llm_dat'].append(t_llm_dat)

        hd_h_feat, hd_w_feat = hd_hw
        print(f"  [{i:02d}] {Nq:5d}  {Nq_hd:5d}  {lr_hw[0]}×{lr_hw[1]:<3d}  "
              f"{hd_h_feat}×{hd_w_feat:<4d}"
              f"  {t_dat:>8.1f}  {t_fair:>8.1f}  {t_lr:>8.1f}"
              f"  {dat_vs_fair:>9.2f}x"
              f"  {llm_dat_ovhd:>+11.1f}  {llm_fair_ovhd:>+14.1f}")

    if not results['dat']:
        print("  没有成功处理的样本，跳过 Part B")
        return

    def avg(k): return sum(results[k]) / len(results[k])
    aN       = len(results['dat'])
    a_dat    = avg('dat');  a_fair = avg('fair');  a_lr = avg('lr')
    a_hd     = avg('hd_enc')
    a_llm_dat = avg('llm_dat')
    a_nq_lr  = sum(nq_lrs) / aN
    a_nq_hd  = sum(nq_hds) / aN

    # 时间分解：
    #   LR-only  E2E = LR ViT + LLM(Nq_lr)
    #   Fair-HD  E2E = HD ViT + LLM(Nq_hd)
    #   DAT      E2E = LR ViT + HD ViT + LLM-DAT(Nq_lr, cross-attn)
    #   LR ViT 估算 = DAT E2E - HD ViT - LLM-DAT-only
    vit_lr_est    = a_dat - a_hd - a_llm_dat   # LR ViT 时间估算
    llm_lr_est    = a_lr - vit_lr_est           # LLM-only LR 时间估算
    llm_fair_only = a_fair - a_hd               # LLM-only Fair-HD（≈ Fair-HD - HD ViT）

    print(f"\n{'─'*116}")
    print(f"  样本数: {aN}  平均 Nq-LR={a_nq_lr:.0f}  Nq-HD={a_nq_hd:.0f}"
          f"  (HD序列比LR长 {(a_nq_hd/a_nq_lr - 1)*100:.0f}%，"
          f"HD token数比LR多 {(a_nq_hd - a_nq_lr):.0f})")
    print()
    print(f"  ┌─── E2E Prefill 延迟（含 ViT 编码）───────────────────────────")
    print(f"  │  LR-only  : {a_lr:>7.1f} ms  (LR ViT + LLM Nq={a_nq_lr:.0f}，无 HD，速度下界)")
    print(f"  │  Fair-HD  : {a_fair:>7.1f} ms  (HD ViT + LLM Nq={a_nq_hd:.0f}，HD 直接作为 tokens）")
    print(f"  │  DAT      : {a_dat:>7.1f} ms  (LR ViT + HD ViT + LLM Nq={a_nq_lr:.0f}，cross-attn)")
    print(f"  │  Fair-HD vs LR-only : {a_fair - a_lr:>+7.1f} ms  ({(a_fair/a_lr - 1)*100:+.1f}%)")
    print(f"  │  DAT      vs LR-only: {a_dat  - a_lr:>+7.1f} ms  ({(a_dat /a_lr - 1)*100:+.1f}%)")
    print(f"  │  DAT      vs Fair-HD: {a_dat/a_fair:.2f}x  "
          f"({'慢' if a_dat > a_fair else '快'} {abs(a_dat - a_fair):.1f} ms)")
    print(f"  │")
    print(f"  ├─── ViT 编码拆分 ──────────────────────────────────────────────")
    print(f"  │  HD ViT (单独测量)  : {a_hd:>7.1f} ms")
    print(f"  │  LR ViT (估算)      : {vit_lr_est:>7.1f} ms  (DAT E2E - HD ViT - LLM-DAT)")
    print(f"  │  HD/LR ViT 比       : {a_hd/vit_lr_est:.2f}x")
    print(f"  │")
    print(f"  └─── LLM-only 开销对比（排除 ViT）────────────────────────────")
    print(f"       LLM LR-only  (Nq={a_nq_lr:.0f}): {llm_lr_est:>7.1f} ms  (参考，估算)")
    print(f"       LLM Fair-HD  (Nq={a_nq_hd:.0f}): {llm_fair_only:>7.1f} ms  (≈ Fair-HD - HD ViT，估算)")
    print(f"       LLM DAT-only (Nq={a_nq_lr:.0f}): {a_llm_dat:>7.1f} ms  (cross-attn，直接测量)")
    print(f"       LLM Fair-HD vs LR-only : {llm_fair_only - llm_lr_est:>+7.1f} ms"
          f"  ({(llm_fair_only/llm_lr_est - 1)*100:+.1f}%，序列长 {a_nq_hd/a_nq_lr:.1f}x)")
    print(f"       LLM DAT     vs LR-only : {a_llm_dat - llm_lr_est:>+7.1f} ms"
          f"  ({(a_llm_dat/llm_lr_est - 1)*100:+.1f}%，DAT cross-attn 开销)")
    print(f"       DAT LLM 效率优势        : {llm_fair_only / a_llm_dat:.1f}x  (Fair-HD LLM / DAT LLM)")
    print(f"{'─'*116}")


# ════════════════════════════════════════════════════════════════════════════════
# Part C：逐层耗时分布（CUDA event hook）
# ════════════════════════════════════════════════════════════════════════════════
def run_layerwise_benchmark(dat_model, processor, vstar_samples,
                            n_iters=5, device=DEVICE, dtype=DTYPE):
    """
    用 CUDA event hook 逐层对比两种 HD 集成方案：
      · DAT     : Nq_lr 序列 + HD cross-attention（6 DAT 层有额外开销）
      · Fair-HD : Nq_hd 序列 + 标准因果 attention（更长序列 → 每层均更慢）
    """
    print(f"\n{'═'*116}")
    print(f"Part C — 逐层耗时分布（DAT(Nq_lr) vs Fair-HD(Nq_hd)，CUDA event hook，{n_iters} iters）")
    print(f"  DAT 层索引: {DAT_LAYER_INDICES}  (共 {len(DAT_LAYER_INDICES)} 层)")
    print(f"{'═'*116}")

    # 找第一个有效样本
    inputs_lr = inputs_hd = inputs_hd_full = None
    for sample in vstar_samples:
        img_path = os.path.join(VSTAR_DIR, sample['image'])
        if not os.path.exists(img_path):
            continue
        try:
            inputs_lr, inputs_hd, _, _, inputs_hd_full = process_sample(
                processor, img_path, sample['text'], device=device)
            break
        except Exception:
            continue

    if inputs_lr is None:
        print("  没有可用样本，跳过 Part C")
        return

    pv_hd     = inputs_hd['pixel_values'].to(dtype)
    thw_hd    = inputs_hd['image_grid_thw']
    pv_lr     = inputs_lr['pixel_values'].to(dtype)
    thw_lr    = inputs_lr['image_grid_thw']
    iids      = inputs_lr['input_ids']
    amask     = inputs_lr.get('attention_mask')
    Nq        = iids.shape[1]
    pv_hdf    = inputs_hd_full['pixel_values'].to(dtype)
    thw_hdf   = inputs_hd_full['image_grid_thw']
    iids_hdf  = inputs_hd_full['input_ids']
    amask_hdf = inputs_hd_full.get('attention_mask')
    Nq_hd     = iids_hdf.shape[1]

    with torch.no_grad():
        hd_feats = dat_model._generate_hd_features(pv_hd, thw_hd)

    def fwd_dat():
        with torch.no_grad():
            dat_model(input_ids=iids, attention_mask=amask,
                      pixel_values=pv_lr, image_grid_thw=thw_lr,
                      image_hd_features=hd_feats, use_cache=False)

    def fwd_fair_hd():
        with torch.no_grad():
            dat_model(input_ids=iids_hdf, attention_mask=amask_hdf,
                      pixel_values=pv_hdf, image_grid_thw=thw_hdf,
                      pixel_values_hd=None, image_grid_thw_hd=None,
                      use_cache=False)

    # warmup
    with torch.no_grad():
        for _ in range(3):
            fwd_dat()
            fwd_fair_hd()
    torch.cuda.synchronize()

    print(f"  测量中（DAT Nq={Nq}，Fair-HD Nq={Nq_hd}）…", end="", flush=True)
    t_dat  = layerwise_timing(dat_model, fwd_dat,     n_iters=n_iters)
    t_fair = layerwise_timing(dat_model, fwd_fair_hd, n_iters=n_iters)
    print(" done")

    N           = len(dat_model.model.language_model.layers)
    total_dat   = sum(t_dat.values())
    total_fair  = sum(t_fair.values())
    dat_set     = set(DAT_LAYER_INDICES)
    std_set     = set(range(N)) - dat_set

    print(f"\n  {'层':>4}  {'类型':>5}  {'DAT(ms)':>9}  {'FairHD(ms)':>11}"
          f"  {'开销(ms)':>9}  {'开销%':>8}  {'DAT占总%':>8}")
    print(f"  {'─'*72}")

    for i in range(N):
        layer_type = 'DAT' if i in dat_set else 'Std'
        td = t_dat.get(i, 0.0); tf = t_fair.get(i, 0.0)
        diff = td - tf
        pct_ovhd = diff / tf * 100 if tf > 0 else 0
        pct_tot  = td / total_dat * 100
        marker   = ' ◄' if i in dat_set else ''
        print(f"  {i:>4}  {layer_type:>5}  {td:>9.2f}  {tf:>11.2f}"
              f"  {diff:>+9.2f}  {pct_ovhd:>+7.1f}%  {pct_tot:>7.1f}%{marker}")

    print(f"  {'─'*72}")
    print(f"  {'合计':>4}  {'':>5}  {total_dat:>9.2f}  {total_fair:>11.2f}"
          f"  {total_dat - total_fair:>+9.2f}"
          f"  {(total_dat - total_fair) / total_fair * 100:>+7.1f}%")

    # 分组统计
    avg_dat_d   = sum(t_dat[i]  for i in dat_set) / len(dat_set)
    avg_fair_d  = sum(t_fair[i] for i in dat_set) / len(dat_set)
    avg_dat_s   = sum(t_dat[i]  for i in std_set) / len(std_set)
    avg_fair_s  = sum(t_fair[i] for i in std_set) / len(std_set)

    print(f"\n  平均每 DAT 层（{len(dat_set)} 层）  :"
          f" DAT={avg_dat_d:.2f} ms  vs  FairHD={avg_fair_d:.2f} ms"
          f"  (DAT开销 {avg_dat_d - avg_fair_d:+.2f} ms /"
          f" {(avg_dat_d - avg_fair_d) / avg_fair_d * 100:+.1f}%)")
    print(f"  平均每标准层（{len(std_set)} 层） :"
          f" DAT={avg_dat_s:.2f} ms  vs  FairHD={avg_fair_s:.2f} ms"
          f"  (DAT开销 {avg_dat_s - avg_fair_s:+.2f} ms /"
          f" {(avg_dat_s - avg_fair_s) / avg_fair_s * 100:+.1f}%)")
    print(f"  注：FairHD 每层均因 Nq={Nq_hd}>{Nq} 而更慢；DAT 层额外含 HD cross-attn 但受益于短序列")
    print(f"{'─'*116}")


# ════════════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════════════
def main():
    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        convert_qwen2_5vl_to_dat, _FA_BACKEND,
    )

    print(f"\nPyTorch   : {torch.__version__}")
    print(f"CUDA      : {torch.version.cuda}")
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"FA backend: {_FA_BACKEND}")
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
            inputs_lr, _, lr_hw, hd_hw, _ = process_sample(
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
    print(f"  DAT = LR ViT + HD ViT + LLM(Nq_lr)")
    print(f"  Fair-HD = HD ViT + LLM(Nq_hd)")
    print(f"  LR-only = LR ViT + LLM(Nq_lr)")
    print(f"{'═'*72}")
    for sample in vstar_samples:
        img_path = os.path.join(VSTAR_DIR, sample['image'])
        if not os.path.exists(img_path):
            continue
        try:
            inputs_lr, inputs_hd, _, _, inputs_hd_full = process_sample(
                processor, img_path, sample['text'], device=DEVICE)
            pv_hd    = inputs_hd['pixel_values'].to(DTYPE)
            thw_hd   = inputs_hd['image_grid_thw']
            pv_lr    = inputs_lr['pixel_values'].to(DTYPE)
            thw_lr   = inputs_lr['image_grid_thw']
            iids     = inputs_lr['input_ids']
            amask    = inputs_lr.get('attention_mask')
            pv_hdf   = inputs_hd_full['pixel_values'].to(DTYPE)
            thw_hdf  = inputs_hd_full['image_grid_thw']
            iids_hdf = inputs_hd_full['input_ids']
            amask_hdf = inputs_hd_full.get('attention_mask')

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
            mem_fair = measure_peak(lambda: dat_model(
                input_ids=iids_hdf, attention_mask=amask_hdf,
                pixel_values=pv_hdf, image_grid_thw=thw_hdf,
                pixel_values_hd=None, image_grid_thw_hd=None,
                use_cache=False,
            ))
            mem_lr = measure_peak(lambda: dat_model(
                input_ids=iids, attention_mask=amask,
                pixel_values=pv_lr, image_grid_thw=thw_lr,
                pixel_values_hd=None, image_grid_thw_hd=None,
                use_cache=False,
            ))
            Nq_lr_m  = iids.shape[1]
            Nq_hd_m  = iids_hdf.shape[1]
            print(f"  Nq_lr={Nq_lr_m}  Nq_hd={Nq_hd_m}")
            print(f"  Peak DAT     : {mem_dat:.0f} MB  (LR ViT + HD ViT + LLM-DAT)")
            print(f"  Peak Fair-HD : {mem_fair:.0f} MB  (HD ViT + LLM-FairHD)")
            print(f"  Peak LR-only : {mem_lr:.0f} MB  (LR ViT + LLM-LR)")
            print(f"  DAT vs LR-only  : {mem_dat - mem_lr:+.0f} MB")
            print(f"  Fair-HD vs LR-only: {mem_fair - mem_lr:+.0f} MB")
            break
        except Exception as e:
            print(f"  [skip] {e}")
            continue

    print("\n✓ 测速完成")


if __name__ == '__main__':
    main()
