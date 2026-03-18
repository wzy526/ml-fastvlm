"""test_dat_flops.py

DAT Two-pass + LSE merge FLOPs 测试。

测量维度：
  1. 理论 FLOPs（解析公式）
  2. 实测 FLOPs（torch.utils.flop_counter.FlopCounterMode）
  3. 实测耗时（CUDA events）

对比方案：
  A. Baseline  — 标准因果 attention（无 HD），仅 Pass 1
  B. Two-pass  — Pass 1 + Pass 2(HD cross-attn) + LSE merge（本实现）
  C. ExtKV     — 旧方案：把 HD 拼入 KV 后做 [Nq × (Nq+Lp*Ns)] attention（理论基准）

Run:
    python test_dat_flops.py
    python test_dat_flops.py --Nq 1024 --Nans 400 --Lp 4 --Ns 144
"""

import sys, argparse, math, time
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, "/root/ml-fastvlm")
from llava.model.language_model.modeling_qwen2_5vl_dat import (
    _dat_attn_with_lse,
    _FLASH_ATTN_AVAILABLE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def flops_sdpa(Nq: int, Nk: int, H: int, D: int) -> int:
    """FLOPs for a single-batch attention:  2 × H × Nq × Nk × D  (QK^T + AV)."""
    return 2 * H * Nq * Nk * D


def hr(f: float) -> str:
    """Human-readable FLOP string."""
    if f >= 1e12:
        return f"{f/1e12:.3f} TFLOPs"
    if f >= 1e9:
        return f"{f/1e9:.3f} GFLOPs"
    return f"{f/1e6:.3f} MFLOPs"


def measure_flops(fn, *args) -> int:
    """Measure FLOPs with FlopCounterMode (matmul / linear only)."""
    with FlopCounterMode(display=False) as fc:
        fn(*args)
    return fc.get_total_flops()


def cuda_time_ms(fn, *args, warmup=3, repeat=10) -> float:
    """Average CUDA kernel time in ms."""
    device = args[0].device
    if device.type != "cuda":
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn(*args)
        return (time.perf_counter() - t0) / repeat * 1000

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeat):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


# ─────────────────────────────────────────────────────────────────────────────
# Attention functions under test
# ─────────────────────────────────────────────────────────────────────────────
def run_baseline(Q, K, V):
    """Pass 1 only: standard causal attention."""
    out, lse = _dat_attn_with_lse(Q, K, V, causal=True)
    return out


def run_two_pass(Q, K, V, K_hd_list, V_hd_list, ans_ranges):
    """Full two-pass: Pass 1 + per-segment HD cross-attn + LSE merge."""
    out1, lse1 = _dat_attn_with_lse(Q, K, V, causal=True)

    # Accumulate per-batch — here B=1, single batch element
    out_b = out1[0:1]   # view
    for (ans_start, ans_end), K_hd, V_hd in zip(ans_ranges, K_hd_list, V_hd_list):
        Nans = ans_end - ans_start
        q_ans = Q[0:1, :, ans_start:ans_end, :]      # [1, H, Nans, D]
        out2, lse2 = _dat_attn_with_lse(q_ans, K_hd, V_hd, causal=False)

        # LSE merge (element-wise — cheap)
        lse1_ans = lse1[0:1, :, ans_start:ans_end]
        lse  = torch.logaddexp(lse1_ans, lse2)
        w1   = (lse1_ans - lse).exp().permute(0, 2, 1).unsqueeze(-1)
        w2   = (lse2     - lse).exp().permute(0, 2, 1).unsqueeze(-1)
        out_b_new       = out_b.clone()
        out1_ans        = out_b[:, ans_start:ans_end, :, :]
        out_b_new[:, ans_start:ans_end, :, :] = w1 * out1_ans + w2 * out2
        out_b = out_b_new

    return out_b


def run_extkv(Q, K_ext, V_ext, Nq):
    """Extended-KV (old approach): attend over [K; K_hd] with mask."""
    B, H, Nk, D = K_ext.shape
    scale = D ** -0.5
    scores = torch.matmul(Q, K_ext.transpose(-1, -2)) * scale   # [B,H,Nq,Nk]
    # Causal over original Nq, full visibility for HD part (simplified)
    causal_mask = torch.zeros(1, 1, Nq, Nk, device=Q.device, dtype=torch.bool)
    causal_mask[:, :, :, :Nq] = torch.triu(
        torch.ones(Nq, Nq, device=Q.device, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float('-inf'))
    attn_w = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
    return torch.matmul(attn_w, V_ext).transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nq",   type=int, default=490,  help="sequence length")
    parser.add_argument("--H",    type=int, default=16,   help="num attention heads")
    parser.add_argument("--D",    type=int, default=128,  help="head dim")
    parser.add_argument("--Ns",   type=int, default=144,  help="HD tokens per segment (grid²)")
    parser.add_argument("--Lp",   type=int, default=5,    help="number of answer segments")
    parser.add_argument("--Nans", type=int, default=None, help="total answer tokens (default Nq//4)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype  = torch.float32
    B, H, Nq, D, Ns, Lp = 1, args.H, args.Nq, args.D, args.Ns, args.Lp
    total_Nans = args.Nans if args.Nans else Nq // 4
    # Distribute answer tokens evenly across Lp segments
    base_nans  = total_Nans // Lp
    Nans_list  = [base_nans] * (Lp - 1) + [total_Nans - base_nans * (Lp - 1)]

    # Build answer ranges (evenly spaced after the first half of the sequence)
    ans_start_base = Nq // 2
    ans_ranges = []
    cursor = ans_start_base
    for n in Nans_list:
        ans_ranges.append((cursor, cursor + n))
        cursor += n + 5   # small gap between segments

    print(f"\n{'='*70}")
    print(f"  DAT FLOPs Test  —  device={device}  flash_attn={'yes' if _FLASH_ATTN_AVAILABLE else 'no (sdpa fallback)'}")
    print(f"{'='*70}")
    print(f"  B={B}  H={H}  Nq={Nq}  D={D}  Ns={Ns}  Lp={Lp}  total_Nans={total_Nans}")
    print(f"  Nans_per_seg={Nans_list}")
    print(f"  ans_ranges  ={ans_ranges}")

    # ── Tensors ──────────────────────────────────────────────────────────────
    torch.manual_seed(0)
    Q   = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
    K   = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
    V   = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
    K_hd_list = [torch.randn(1, H, Ns, D, device=device, dtype=dtype) for _ in range(Lp)]
    V_hd_list = [torch.randn(1, H, Ns, D, device=device, dtype=dtype) for _ in range(Lp)]

    # Extended-KV for old approach
    K_ext = torch.cat([K] + K_hd_list, dim=2)   # [B, H, Nq+Lp*Ns, D]
    V_ext = torch.cat([V] + V_hd_list, dim=2)
    Nk_ext = Nq + Lp * Ns

    # ── Theoretical FLOPs ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  THEORETICAL FLOPs  (QK^T + AttnV per attention block, B=1)")
    print(f"{'─'*70}")

    flops_pass1 = flops_sdpa(Nq, Nq, H, D)
    flops_pass2 = sum(flops_sdpa(n, Ns, H, D) for n in Nans_list)
    flops_two_pass = flops_pass1 + flops_pass2

    flops_base   = flops_sdpa(Nq, Nq, H, D)
    flops_extkv  = flops_sdpa(Nq, Nk_ext, H, D)

    print(f"  Baseline (standard causal, no HD)  : {hr(flops_base)}")
    print(f"  Two-pass Pass1 (causal Nq×Nq)      : {hr(flops_pass1)}")
    print(f"  Two-pass Pass2 (HD cross-attn ×Lp) : {hr(flops_pass2)}")
    print(f"  Two-pass TOTAL                     : {hr(flops_two_pass)}")
    print(f"  ExtKV (old, Nq×{Nk_ext})       : {hr(flops_extkv)}")
    print()
    print(f"  Two-pass overhead vs Baseline      : +{hr(flops_pass2)}  "
          f"({flops_pass2/flops_base*100:.1f}% of Pass1)")
    print(f"  Two-pass saving vs ExtKV           : -{hr(flops_extkv - flops_two_pass)}  "
          f"({(flops_extkv-flops_two_pass)/flops_extkv*100:.1f}% reduction)")

    # ── Measured FLOPs (FlopCounterMode) ────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  MEASURED FLOPs  (torch.utils.flop_counter, matmul/linear ops)")
    print(f"{'─'*70}")

    mf_baseline  = measure_flops(run_baseline, Q, K, V)
    mf_two_pass  = measure_flops(run_two_pass, Q, K, V, K_hd_list, V_hd_list, ans_ranges)
    mf_extkv     = measure_flops(run_extkv, Q, K_ext, V_ext, Nq)

    print(f"  Baseline  : {hr(mf_baseline)}")
    print(f"  Two-pass  : {hr(mf_two_pass)}")
    print(f"  ExtKV     : {hr(mf_extkv)}")
    print(f"  Two-pass overhead  : +{hr(mf_two_pass - mf_baseline)}  "
          f"({(mf_two_pass-mf_baseline)/mf_baseline*100:.1f}%)")
    print(f"  Two-pass vs ExtKV  : {(mf_two_pass/mf_extkv*100):.1f}% of ExtKV FLOPs")

    # ── Wall-clock timing ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  WALL-CLOCK  (avg over 10 runs, warmup=3,  device={device})")
    print(f"{'─'*70}")

    t_baseline = cuda_time_ms(run_baseline, Q, K, V)
    t_two_pass = cuda_time_ms(run_two_pass, Q, K, V, K_hd_list, V_hd_list, ans_ranges)
    t_extkv    = cuda_time_ms(run_extkv, Q, K_ext, V_ext, Nq)

    print(f"  Baseline  : {t_baseline:.3f} ms")
    print(f"  Two-pass  : {t_two_pass:.3f} ms  (overhead +{t_two_pass-t_baseline:.3f} ms)")
    print(f"  ExtKV     : {t_extkv:.3f} ms")
    print(f"  Two-pass speedup vs ExtKV : {t_extkv/t_two_pass:.2f}×")

    # ── Scaling sweep: vary Nq ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  SCALING  —  theoretical FLOPs vs Nq  (Lp=5, Nans=Nq/4, Ns=144)")
    print(f"{'─'*70}")
    print(f"  {'Nq':>6}  {'Nk_ext':>8}  {'Baseline':>12}  {'Two-pass':>12}  {'ExtKV':>12}  {'ratio(TP/EK)':>12}")
    for nq in [256, 490, 768, 1024, 2048, 4096]:
        nans_total = nq // 4
        nans_per   = nans_total // Lp
        nans_list_ = [nans_per] * (Lp - 1) + [nans_total - nans_per * (Lp - 1)]
        nk_ext_    = nq + Lp * Ns
        f_base_    = flops_sdpa(nq, nq, H, D)
        f_tp_      = flops_sdpa(nq, nq, H, D) + sum(flops_sdpa(n, Ns, H, D) for n in nans_list_)
        f_ek_      = flops_sdpa(nq, nk_ext_, H, D)
        print(f"  {nq:>6}  {nk_ext_:>8}  {hr(f_base_):>12}  {hr(f_tp_):>12}  {hr(f_ek_):>12}  {f_tp_/f_ek_*100:>10.1f}%")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
