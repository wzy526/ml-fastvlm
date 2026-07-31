#!/usr/bin/env python3
"""
analyze_vit_llm_flops.py — Qwen2.5-VL ViT vs LLM 参数量 / FLOPs 静态分析 (纯解析计算, 无依赖)。

回答两个问题:
1. ViT 和 LLM 的参数量、prefill FLOPs 对比 (Base 路径 vs DAT 双路)。
2. 各分辨率下 Base / DAT 的输入分辨率与 token 数。

口径说明:
- FLOPs = 2 * MACs; 线性层 2*N*din*dout。
- LLM causal attention 记一半 (FA 只算下三角): 每层 2*S^2*d_attn。
- ViT window-attn 每 token 上下文 = 64 (112px / 14 = 8x8 patch window);
  full-attn block 上下文 = 全序列 N。
- LLM prefill 按 logits_to_keep=1 (lm_head 只算 1 个 token, 可忽略)。
- DAT 层自身的 deformable/gate 额外 FLOPs 很小 (~线性于 S_lr), 不计入, 只影响 <1%。

用法: python scripts/analyze_vit_llm_flops.py
"""

# ── 配置 (来自官方 config.json) ─────────────────────────────────────────────
VIT = dict(depth=32, d=1280, inter=3420, n_full=4, patch=14, merge=2,
           tps=2, in_ch=3, window_tokens=64)  # window 112px -> 8x8 patch

LLM_CFG = {
    '3B': dict(d=2048, layers=36, heads=16, kv=2, head_dim=128,
               inter=11008, vocab=151936, tied=True,  vit_out=2048),
    '7B': dict(d=3584, layers=28, heads=28, kv=4, head_dim=128,
               inter=18944, vocab=152064, tied=False, vit_out=3584),
}

RESOLUTIONS = [672, 1008, 1344, 2016, 2688, 3360]
HR_SCALE = 3
N_TEXT = 30          # chat template + 问题的文本 token, 约数
GIGA = 1e9


# ── 参数量 ──────────────────────────────────────────────────────────────────

def vit_params(out_dim):
    v = VIT
    patch_embed = v['in_ch'] * v['tps'] * v['patch'] ** 2 * v['d']
    per_block = (4 * v['d'] ** 2 + 4 * v['d']          # qkv(+bias) + proj
                 + 3 * v['d'] * v['inter']             # SwiGLU gate/up/down
                 + 2 * v['d'])                         # 2x RMSNorm
    merge_dim = v['d'] * v['merge'] ** 2               # 5120
    merger = merge_dim ** 2 + merge_dim + merge_dim * out_dim + out_dim + v['d']
    return patch_embed + v['depth'] * per_block + merger


def llm_params(c):
    d, hd = c['d'], c['head_dim']
    attn = (d * c['heads'] * hd + c['heads'] * hd          # q + bias
            + 2 * (d * c['kv'] * hd + c['kv'] * hd)        # k,v + bias
            + c['heads'] * hd * d)                         # o
    mlp = 3 * d * c['inter']
    per_layer = attn + mlp + 2 * d
    embed = c['vocab'] * d * (1 if c['tied'] else 2)
    return c['layers'] * per_layer + embed + d


# ── FLOPs ───────────────────────────────────────────────────────────────────

def vit_flops(R, out_dim):
    """单张 R×R 图过一次 ViT 的 FLOPs。"""
    v = VIT
    N = (R // v['patch']) ** 2
    f_patch = 2 * N * (v['in_ch'] * v['tps'] * v['patch'] ** 2) * v['d']
    f_lin_blk = 2 * N * (4 * v['d'] ** 2 + 3 * v['d'] * v['inter'])
    f_attn_win = 4 * N * v['window_tokens'] * v['d']
    f_attn_full = 4 * N * N * v['d']
    n_win = v['depth'] - v['n_full']
    merge_dim = v['d'] * v['merge'] ** 2
    f_merger = 2 * (N // 4) * (merge_dim ** 2 + merge_dim * out_dim)
    total = (f_patch + v['depth'] * f_lin_blk
             + n_win * f_attn_win + v['n_full'] * f_attn_full + f_merger)
    return total, N, dict(linear=v['depth'] * f_lin_blk + f_patch + f_merger,
                          win_attn=n_win * f_attn_win,
                          full_attn=v['n_full'] * f_attn_full)


def llm_prefill_flops(S, c):
    """S 个 token 的 causal prefill FLOPs (lm_head 忽略)。"""
    d, d_attn = c['d'], c['heads'] * c['head_dim']
    lin = 2 * S * (d * d_attn + 2 * d * c['kv'] * c['head_dim']
                   + d_attn * d + 3 * d * c['inter'])
    attn = 2 * S * S * d_attn        # causal: 4*S^2*d 的一半
    return c['layers'] * (lin + attn)


# ── 报表 ────────────────────────────────────────────────────────────────────

def fmt(x):
    return f"{x / GIGA:8.1f}"


def main():
    for size, c in LLM_CFG.items():
        p_vit, p_llm = vit_params(c['vit_out']), llm_params(c)
        print(f"\n{'=' * 100}")
        print(f"Qwen2.5-VL-{size}:  ViT {p_vit/1e6:.0f}M  |  LLM {p_llm/1e9:.2f}B"
              f"  |  ViT/LLM 参数比 {p_vit/p_llm*100:.1f}%")
        print(f"{'=' * 100}")

        hdr = (f"{'R(HD)':>6} {'LR':>5} | {'ViT_tok HD':>10} {'LR':>7} "
               f"| {'LLM_tok Base':>12} {'DAT':>6} "
               f"| {'GF ViT_HD':>9} {'ViT_LR':>7} {'LLM_B':>8} {'LLM_D':>8} "
               f"| {'Base总':>8} {'DAT总':>8} {'ViT占B':>6} {'ViT占D':>6}")
        print(hdr)
        print('-' * len(hdr))

        for R in RESOLUTIONS:
            lr = R // HR_SCALE
            f_hd, n_hd, _ = vit_flops(R, c['vit_out'])
            f_lr, n_lr, _ = vit_flops(lr, c['vit_out'])
            s_base = (R // 28) ** 2 + N_TEXT
            s_dat = (lr // 28) ** 2 + N_TEXT
            f_llm_b = llm_prefill_flops(s_base, c)
            f_llm_d = llm_prefill_flops(s_dat, c)
            base_total = f_hd + f_llm_b
            dat_total = f_hd + f_lr + f_llm_d
            print(f"{R:>6} {lr:>5} | {n_hd:>10} {n_lr:>7} "
                  f"| {s_base - N_TEXT:>12} {s_dat - N_TEXT:>6} "
                  f"| {fmt(f_hd)} {fmt(f_lr)[:7]:>7} {fmt(f_llm_b)} {fmt(f_llm_d)} "
                  f"| {fmt(base_total)} {fmt(dat_total)} "
                  f"{f_hd/base_total*100:>5.0f}% {(f_hd+f_lr)/dat_total*100:>5.0f}%")

        # ViT 内部 full-attn 占比
        print(f"\n  ViT(HD) 内部分解 (GFLOPs): linear / window-attn / full-attn")
        for R in RESOLUTIONS:
            f, n, brk = vit_flops(R, c['vit_out'])
            print(f"    R={R:>5} (tok={n:>6}): {brk['linear']/GIGA:>8.1f} / "
                  f"{brk['win_attn']/GIGA:>6.1f} / {brk['full_attn']/GIGA:>8.1f}"
                  f"   (full-attn 占 ViT {brk['full_attn']/f*100:.0f}%)")


if __name__ == '__main__':
    main()
