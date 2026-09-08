#!/usr/bin/env python3
"""Numerical check of _TwoPassMergedAttnFn (exact merged-gradient two-pass attention).

Compares forward output AND gradients (dq, dk, dv, dk_hd, dv_hd) against an
fp32 eager reference that computes ONE softmax over the concatenated key set
(causal sequence keys + the segment's HD keys), which is what the merged
backward claims to be exactly equal to. Also reports how far the legacy
detached-LSE gradients deviate, so the effect of the fix is visible.

Backend-agnostic: uses whatever raw flash backward the model module picked
(FA2 flash_attn_interface._flash_attn_backward on ordinary pods, FA4
flash_attn.cute._flash_attn_bwd when nvidia-cutlass-dsl is installed). Run
on the training pod:

  python scripts/_test_exact_merge_grad.py                  # auto backend
  DAT_ATTN_BACKEND=fa2 python scripts/_test_exact_merge_grad.py
  DAT_ATTN_BACKEND=fa4 python scripts/_test_exact_merge_grad.py
"""

import math
import sys

import torch

from llava.model.language_model import modeling_qwen3_5_dat as M

if not M._EXACT_MERGE_AVAILABLE:
    sys.exit(f"exact-merge path unavailable here (backend={M._EXACT_BWD_BACKEND}, "
             f"DAT_EXACT_MERGE_GRAD={int(M._EXACT_MERGE_GRAD)}); nothing to test.")
print(f"[test] exact-merge backend: {M._EXACT_BWD_BACKEND.upper()} "
      f"(forward via {'FA4' if M._USE_FA4 else 'FA2 v' + str(M._fa_ver)})")

torch.manual_seed(0)
dev = "cuda"
B, H, N, D = 2, 4, 96, 256           # D=256 matches Qwen3.5 head_dim
# (b, row_start, n_rows, n_hd_keys) — row-disjoint within a batch element
SEGS = [(0, 24, 16, 40), (0, 60, 20, 40), (1, 30, 30, 64)]
scale = 1.0 / math.sqrt(D)


def make():
    q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k2 = [torch.randn(nk, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
          for (_, _, _, nk) in SEGS]
    v2 = [torch.randn(nk, H, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
          for (_, _, _, nk) in SEGS]
    gw = torch.randn(B, N, H, D, device=dev, dtype=torch.float32)
    return q, k, v, k2, v2, gw


def run_exact(q, k, v, k2, v2, gw):
    q2p = torch.cat([q[b, :, s:s + n].transpose(0, 1) for (b, s, n, _) in SEGS], 0)  # [tot,H,D]
    k2p, v2p = torch.cat(k2, 0), torch.cat(v2, 0)
    nq = [n for (_, _, n, _) in SEGS]
    nk = [x for (_, _, _, x) in SEGS]
    cu_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(nq), 0)), dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(nk), 0)), dtype=torch.int32, device=dev)
    seg_meta = tuple((b, s, n) for (b, s, n, _) in SEGS)
    out = M._TwoPassMergedAttnFn.apply(q, k, v, q2p, k2p, v2p, cu_q, cu_k,
                                       max(nq), max(nk), seg_meta, None)   # [B,N,H,D]
    (out.float() * gw).sum().backward()
    return out.detach().float(), [t.grad.float() for t in (q, k, v)], \
        [t.grad.float() for t in k2], [t.grad.float() for t in v2]


def run_legacy(q, k, v, k2, v2, gw):
    """Detached-LSE merge (historical FA2 semantics) via the module helpers."""
    out1, lse1 = M._dat_attn_with_lse(q, k, v, causal=True)
    seg_q = [q[b, :, s:s + n].transpose(0, 1).contiguous() for (b, s, n, _) in SEGS]
    out2_list, lse2_list = M._dat_cross_attn_varlen(seg_q, list(k2), list(v2))
    out = out1.clone()
    for i, (b, s, n, _) in enumerate(SEGS):
        l1 = lse1[b:b + 1, :, s:s + n].float()
        l2 = lse2_list[i].float()
        lm = torch.logaddexp(l1, l2)
        w1 = (l1 - lm).exp().permute(0, 2, 1).unsqueeze(-1)
        w2 = (l2 - lm).exp().permute(0, 2, 1).unsqueeze(-1)
        out[b:b + 1, s:s + n] = (w1 * out1[b:b + 1, s:s + n] + w2 * out2_list[i]).to(out.dtype)
    (out.float() * gw).sum().backward()
    return out.detach().float(), [t.grad.float() for t in (q, k, v)], \
        [t.grad.float() for t in k2], [t.grad.float() for t in v2]


def run_reference(q, k, v, k2, v2, gw):
    """fp32 eager: one softmax per row over [causal seq keys | own-segment HD keys]."""
    qf, kf, vf = (t.float() for t in (q, k, v))
    k2f = [t.float() for t in k2]
    v2f = [t.float() for t in v2]
    outs = []
    for b in range(B):
        segs_b = [(i, s, n, nk) for i, (bb, s, n, nk) in enumerate(SEGS) if bb == b]
        K = torch.cat([kf[b]] + [k2f[i].transpose(0, 1) for (i, _, _, _) in segs_b], dim=1)  # [H, N+ΣNk, D]
        V = torch.cat([vf[b]] + [v2f[i].transpose(0, 1) for (i, _, _, _) in segs_b], dim=1)
        S = torch.einsum("hnd,hmd->hnm", qf[b], K) * scale                                # [H, N, M]
        mask = torch.zeros(N, K.shape[1], dtype=torch.bool, device=dev)
        mask[:, :N] = torch.tril(torch.ones(N, N, dtype=torch.bool, device=dev))
        off = N
        for (i, s, n, nk) in segs_b:
            mask[s:s + n, off:off + nk] = True
            off += nk
        S = S.masked_fill(~mask, float("-inf"))
        P = S.softmax(-1)
        outs.append(torch.einsum("hnm,hmd->hnd", P, V).transpose(0, 1))                   # [N, H, D]
    out = torch.stack(outs, 0)
    (out * gw).sum().backward()
    return out.detach(), [t.grad.float() for t in (q, k, v)], \
        [t.grad.float() for t in k2], [t.grad.float() for t in v2]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def report(name, res, ref):
    out, g, gk2, gv2 = res
    rout, rg, rgk2, rgv2 = ref
    print(f"\n[{name}]  rel err vs fp32 eager reference")
    print(f"  out       : {rel(out, rout):.3e}")
    for nm, a, r in zip(("dq", "dk", "dv"), g, rg):
        print(f"  {nm:<9} : {rel(a, r):.3e}")
    for i in range(len(SEGS)):
        print(f"  dk_hd[{i}]  : {rel(gk2[i], rgk2[i]):.3e}    dv_hd[{i}]  : {rel(gv2[i], rgv2[i]):.3e}")
    return max([rel(out, rout)] + [rel(a, r) for a, r in zip(g, rg)]
               + [rel(gk2[i], rgk2[i]) for i in range(len(SEGS))]
               + [rel(gv2[i], rgv2[i]) for i in range(len(SEGS))])


def fresh():
    torch.manual_seed(0)          # identical values for every run
    return make()


ref = run_reference(*fresh())
exact = run_exact(*fresh())
legacy = run_legacy(*fresh())

worst_exact = report("EXACT merged-stats backward", exact, ref)
worst_legacy = report("LEGACY detached-LSE backward", legacy, ref)

TOL = 3e-2   # bf16 kernels vs fp32 eager
print(f"\nworst rel err: exact={worst_exact:.3e}  legacy={worst_legacy:.3e}  (tol {TOL})")
if worst_exact > TOL:
    sys.exit("FAIL: exact-merge gradients deviate from the concatenated-KV reference")
print("PASS: exact-merge forward+backward match single-softmax-over-union reference")
if worst_legacy <= TOL:
    print("note: legacy grads also within tol on this random instance — the two paths differ "
          "only through dlse terms, which are small when w_hd is small; the exactness claim "
          "is still verified above.")
