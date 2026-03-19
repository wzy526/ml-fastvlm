"""
Benchmark _dat_attn_with_lse: flash_attn vs manual SDPA fallback.

Tests both Pass 1 (causal) and Pass 2 (cross-attention) with realistic
sequence lengths for LLaVA-665K training.
"""

import torch
import time
import sys

# ── Import DAT attention function ─────────────────────────────────────────────
sys.path.insert(0, '/root/ml-fastvlm')
from llava.model.language_model.modeling_qwen2_5vl_dat import (
    _dat_attn_with_lse,
    _FLASH_ATTN_AVAILABLE,
)
_FA_HAS_SOFTMAX_LSE = False  # actual file uses return_attn_probs path

# ── Manual SDPA fallback (always available, for comparison) ───────────────────
def _manual_attn_with_lse(q, k, v, causal=False):
    B, H, Nq, D = q.shape
    scale = D ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        mask = torch.triu(torch.ones(Nq, k.size(2), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    lse = torch.logsumexp(scores.float(), dim=-1)
    attn_w = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn_w, v).transpose(1, 2)
    return out, lse


def benchmark(fn, *args, warmup=5, iters=20, label=""):
    # warmup
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    print(f"  {label:40s}: {elapsed:7.2f} ms/iter")
    return elapsed


def run_case(label, B, H, Nq, Nk, D, dtype, causal, device="cuda"):
    print(f"\n{'='*60}")
    print(f"Case: {label}")
    print(f"  B={B}, H={H}, Nq={Nq}, Nk={Nk}, D={D}, dtype={dtype}, causal={causal}")
    print(f"  flash_attn available: {_FLASH_ATTN_AVAILABLE}, return_softmax_lse: {_FA_HAS_SOFTMAX_LSE}")

    q = torch.randn(B, H, Nq, D, dtype=dtype, device=device)
    k = torch.randn(B, H, Nk, D, dtype=dtype, device=device)
    v = torch.randn(B, H, Nk, D, dtype=dtype, device=device)

    t_flash = None
    t_manual = None

    if _FLASH_ATTN_AVAILABLE:
        try:
            t_flash = benchmark(_dat_attn_with_lse, q, k, v, causal, label="flash_attn (_dat_attn_with_lse)")
        except Exception as e:
            print(f"  flash_attn FAILED: {e}")

    t_manual = benchmark(_manual_attn_with_lse, q, k, v, causal, label="manual SDPA fallback        ")

    if t_flash and t_manual:
        print(f"  → Speedup (manual/flash): {t_manual/t_flash:.2f}x")

    # Memory usage
    torch.cuda.reset_peak_memory_stats()
    if _FLASH_ATTN_AVAILABLE:
        _dat_attn_with_lse(q, k, v, causal)
        mem_flash = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        _manual_attn_with_lse(q, k, v, causal)
        mem_manual = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Memory — flash: {mem_flash:.0f} MB, manual: {mem_manual:.0f} MB")


def main():
    print(f"\nPyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"flash_attn available: {_FLASH_ATTN_AVAILABLE}")
    if _FLASH_ATTN_AVAILABLE:
        try:
            import flash_attn
            print(f"flash_attn version: {flash_attn.__version__}")
        except Exception:
            pass
    print(f"return_softmax_lse: {_FA_HAS_SOFTMAX_LSE}")

    # ── Qwen2.5-VL-7B config ──────────────────────────────────────────────────
    # H=28, D=128 (3584 hidden / 28 heads)
    H  = 28
    D  = 128
    dtype = torch.bfloat16

    # Case 1: Pass 1 (causal), typical training seq (LR tokens + text)
    # LLaVA-665K: ~512 LR tokens + ~256 text ≈ 768
    run_case("Pass 1 — causal, seq=768 (typical single-turn)",
             B=2, H=H, Nq=768,  Nk=768,  D=D, dtype=dtype, causal=True)

    # Case 2: Pass 1 — longer sequence (multi-turn)
    run_case("Pass 1 — causal, seq=1536 (multi-turn)",
             B=2, H=H, Nq=1536, Nk=1536, D=D, dtype=dtype, causal=True)

    # Case 3: Pass 2 (cross-attn), Q=answer tokens, K=HD tokens (Ns=36)
    # grid_size=6 → Ns=36
    run_case("Pass 2 — cross-attn, Nq=256 (ans), Nk=36 (HD, grid=6)",
             B=2, H=H, Nq=256,  Nk=36,   D=D, dtype=dtype, causal=False)

    # Case 4: Pass 2 — larger answer segment
    run_case("Pass 2 — cross-attn, Nq=512 (ans), Nk=36 (HD, grid=6)",
             B=2, H=H, Nq=512,  Nk=36,   D=D, dtype=dtype, causal=False)

    # Case 5: Pass 1 — larger batch
    run_case("Pass 1 — causal, seq=1024, batch=4",
             B=4, H=H, Nq=1024, Nk=1024, D=D, dtype=dtype, causal=True)

    # Case 6: Gradient flow — check backward pass
    print(f"\n{'='*60}")
    print("Backward pass check (flash_attn vs manual)")
    q = torch.randn(1, H, 512, D, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(1, H, 512, D, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(1, H, 512, D, dtype=dtype, device="cuda", requires_grad=True)

    if _FLASH_ATTN_AVAILABLE:
        out_fa, lse_fa = _dat_attn_with_lse(q, k, v, causal=True)
        loss_fa = out_fa.sum() + lse_fa.sum()
        t0 = time.perf_counter()
        loss_fa.backward()
        torch.cuda.synchronize()
        t_bwd_fa = (time.perf_counter() - t0) * 1000
        print(f"  flash_attn backward   : {t_bwd_fa:.2f} ms")
        q.grad = None; k.grad = None; v.grad = None

    out_m, lse_m = _manual_attn_with_lse(q, k, v, causal=True)
    loss_m = out_m.sum() + lse_m.sum()
    t0 = time.perf_counter()
    loss_m.backward()
    torch.cuda.synchronize()
    t_bwd_m = (time.perf_counter() - t0) * 1000
    print(f"  manual SDPA backward  : {t_bwd_m:.2f} ms")

    if _FLASH_ATTN_AVAILABLE:
        print(f"  → Backward speedup: {t_bwd_m/t_bwd_fa:.2f}x")


if __name__ == "__main__":
    main()
