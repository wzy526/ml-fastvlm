#!/usr/bin/env python3
"""Map which GEMM shapes SIGFPE inside cuBLAS on this box (H20-3e hunt).

Each (dtype, M, N, K) case runs in its OWN subprocess, so a SIGFPE kills only
that case and the sweep continues — giving a full crash map in one run.

Shapes covered:
  * lm_head decode/prefill:  M x 2048 -> 151936  (the eval-time crash, M=1)
  * DAT answer-path k/v_proj_hd: M x 2048 -> 256 (the original training crash)

Usage:
    python scripts/test_h20_gemm.py
"""
import subprocess
import sys

CASES = []
for dt in ("bfloat16", "float16", "float32"):
    for M in (1, 2, 3, 4, 8, 16, 32):
        CASES.append((dt, M, 151936, 2048))
for M in (400, 1600, 2000, 8000):
    CASES.append(("bfloat16", M, 256, 2048))

CODE = (
    "import torch;"
    "h=torch.nn.Linear({K},{N},bias=False,dtype=torch.{dt},device='cuda');"
    "x=torch.randn({M},{K},device='cuda',dtype=torch.{dt});"
    "y=h(x);torch.cuda.synchronize()"
)

fails = 0
for dt, M, N, K in CASES:
    r = subprocess.run(
        [sys.executable, "-c", CODE.format(dt=dt, M=M, N=N, K=K)],
        capture_output=True,
    )
    ok = r.returncode == 0
    fails += 0 if ok else 1
    status = "ok" if ok else f"CRASH (rc={r.returncode})"
    print(f"{dt:9s} M={M:<6d} N={N:<7d} K={K:<5d} {status}", flush=True)

print(f"\n{fails}/{len(CASES)} cases crashed")
