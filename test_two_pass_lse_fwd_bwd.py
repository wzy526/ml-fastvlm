"""test_two_pass_lse_fwd_bwd.py

测试两路 attention + LSE 合并的前向、反向数值一致性（fp32）。

测试项：
  T1  _dat_attn_with_lse  前向：causal / non-causal 输出与手写 SDPA 一致
  T2  _dat_attn_with_lse  LSE 值与手写 SDPA 一致
  T3  _merge_two_pass_lse 前向：与联合注意力 Ground Truth 一致
  T4  反向：两路路径的梯度与 Ground Truth 联合注意力梯度一致
  T5  GC 兼容：gradient_checkpointing 包裹后仍能正确 backward

Run:
    python test_two_pass_lse_fwd_bwd.py
"""

import sys
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# ── import actual implementation functions ───────────────────────────────────
sys.path.insert(0, "/root/ml-fastvlm")
from llava.model.language_model.modeling_qwen2_5vl_dat import (
    _dat_attn_with_lse,
    _FLASH_ATTN_AVAILABLE,
)

# ── also reproduce _merge_two_pass_lse as a standalone (mirrors the class method)
def _merge_two_pass_lse(out1, lse1, out2, lse2, ans_start, ans_end):
    """Standalone version of Qwen2_5_VLAttentionDAT._merge_two_pass_lse."""
    out = out1.clone()
    lse1_ans = lse1[:, :, ans_start:ans_end]          # [B, H, Nans]
    out1_ans = out1[:, ans_start:ans_end, :, :]        # [B, Nans, H, D]
    lse = torch.logaddexp(lse1_ans, lse2)              # [B, H, Nans]
    w1 = (lse1_ans - lse).exp().permute(0, 2, 1).unsqueeze(-1)
    w2 = (lse2     - lse).exp().permute(0, 2, 1).unsqueeze(-1)
    out[:, ans_start:ans_end, :, :] = w1 * out1_ans + w2 * out2
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
PASS = "  ✓ PASS"
FAIL = "  ✗ FAIL"

def check(name, actual, ref, atol=1e-5, verbose=True):
    diff = (actual.float() - ref.float()).abs()
    mx   = diff.max().item()
    mn   = diff.mean().item()
    ok   = mx < atol
    tag  = PASS if ok else FAIL
    if verbose:
        print(f"{tag}  {name}")
        if not ok:
            print(f"       max_diff={mx:.3e}  mean_diff={mn:.3e}  atol={atol}")
    return ok


def ref_sdpa(q, k, v, causal=False):
    """Reference hand-written SDPA: returns out [B,Nq,H,D], lse [B,H,Nq]."""
    B, H, Nq, D = q.shape
    scale = D ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale   # [B,H,Nq,Nk]
    if causal:
        cm = torch.triu(torch.ones(Nq, k.size(2), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(cm, float('-inf'))
    lse = torch.logsumexp(scores.float(), dim=-1)            # [B,H,Nq]
    attn_w = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn_w, v).transpose(1, 2)           # [B,Nq,H,D]
    return out, lse


def ground_truth_joint(Q, K, V, K_hd, V_hd, ans_start, ans_end):
    """
    Joint attention with explicit mask:
      K_joint = [K | K_hd],  Q[i] can see K_hd iff i >= ans_start (and i < ans_end for answer-only variant)
    Here we use the precise variant: only answer tokens [ans_start, ans_end) see K_hd.
    """
    B, H, Nq, D = Q.shape
    Ns = K_hd.size(2)
    device = Q.device

    out_list = []
    for b in range(B):
        K_j = torch.cat([K[b:b+1], K_hd[b:b+1]], dim=2)  # [1,H,Nq+Ns,D]
        V_j = torch.cat([V[b:b+1], V_hd[b:b+1]], dim=2)

        scale = D ** -0.5
        scores = (Q[b:b+1] @ K_j.transpose(-1, -2)) * scale  # [1,H,Nq,Nq+Ns]

        # Causal mask over the standard K part
        q_idx = torch.arange(Nq, device=device)
        k_idx = torch.arange(Nq, device=device)
        causal_ok = q_idx.unsqueeze(1) >= k_idx.unsqueeze(0)       # [Nq,Nq]

        # HD mask: only answer tokens see K_hd
        hd_ok = torch.zeros(Nq, Ns, device=device, dtype=torch.bool)
        hd_ok[ans_start:ans_end, :] = True

        full_mask = torch.cat([causal_ok, hd_ok], dim=1)          # [Nq,Nq+Ns]
        scores.masked_fill_(~full_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        out_b = torch.softmax(scores.float(), dim=-1).to(Q.dtype) @ V_j
        out_list.append(out_b.transpose(1, 2))  # [1,Nq,H,D]

    return torch.cat(out_list, dim=0)  # [B,Nq,H,D]


# ─────────────────────────────────────────────────────────────────────────────
# Problem dimensions
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32

B   = 2
H   = 4
D   = 32
Nq  = 24
Ns  = 12   # HD tokens

ANS_START = 8
ANS_END   = 16
Nans      = ANS_END - ANS_START

print(f"Device        : {device}")
print(f"PyTorch       : {torch.__version__}")
print(f"flash_attn    : {'available' if _FLASH_ATTN_AVAILABLE else 'unavailable (using manual fallback)'}")
print(f"Shapes        : B={B} H={H} Nq={Nq} D={D} Ns={Ns} Nans={Nans}")
print()

all_ok = True

# ─────────────────────────────────────────────────────────────────────────────
# T1/T2  _dat_attn_with_lse: forward output and LSE vs ref_sdpa
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("T1/T2  _dat_attn_with_lse  (forward + LSE vs manual SDPA)")
print("=" * 62)

Q1 = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
K1 = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
V1 = torch.randn(B, H, Nq, D, device=device, dtype=dtype)

# causal=True
out_ref_c, lse_ref_c = ref_sdpa(Q1, K1, V1, causal=True)
out_dat_c, lse_dat_c = _dat_attn_with_lse(Q1, K1, V1, causal=True)
all_ok &= check("causal=True  : out matches ref_sdpa", out_dat_c, out_ref_c)
all_ok &= check("causal=True  : lse matches ref_sdpa", lse_dat_c, lse_ref_c)

# causal=False  (non-causal cross-attention)
Q2 = torch.randn(B, H, Nans, D, device=device, dtype=dtype)
K2 = torch.randn(B, H, Ns,   D, device=device, dtype=dtype)
V2 = torch.randn(B, H, Ns,   D, device=device, dtype=dtype)
out_ref_nc, lse_ref_nc = ref_sdpa(Q2, K2, V2, causal=False)
out_dat_nc, lse_dat_nc = _dat_attn_with_lse(Q2, K2, V2, causal=False)
all_ok &= check("causal=False : out matches ref_sdpa", out_dat_nc, out_ref_nc)
all_ok &= check("causal=False : lse matches ref_sdpa", lse_dat_nc, lse_ref_nc)

# ─────────────────────────────────────────────────────────────────────────────
# T3  _merge_two_pass_lse: forward vs ground truth joint attention
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("T3  _merge_two_pass_lse  (forward vs ground-truth joint attn)")
print("=" * 62)

torch.manual_seed(1)
Q  = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
K  = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
V  = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
Kh = torch.randn(B, H, Ns, D, device=device, dtype=dtype)
Vh = torch.randn(B, H, Ns, D, device=device, dtype=dtype)

# Ground truth
out_gt = ground_truth_joint(Q, K, V, Kh, Vh, ANS_START, ANS_END)

# Two-pass
out1, lse1 = _dat_attn_with_lse(Q, K, V, causal=True)   # [B,Nq,H,D], [B,H,Nq]

out_b_list = []
for b in range(B):
    q_ans  = Q[b:b+1, :, ANS_START:ANS_END, :]
    out2_b, lse2_b = _dat_attn_with_lse(q_ans, Kh[b:b+1], Vh[b:b+1], causal=False)
    merged = _merge_two_pass_lse(
        out1[b:b+1], lse1[b:b+1],
        out2_b, lse2_b,
        ANS_START, ANS_END,
    )
    out_b_list.append(merged)

out_two_pass = torch.cat(out_b_list, dim=0)

# Compare only answer tokens (only those attend to HD → should match)
all_ok &= check(
    "answer tokens [ans_start:ans_end] match joint GT",
    out_two_pass[:, ANS_START:ANS_END],
    out_gt[:, ANS_START:ANS_END],
    atol=2e-5,
)
# Non-answer tokens should be unchanged (Pass 1 causal only)
out1_ref_nans = ground_truth_joint(Q, K, V,
    torch.zeros_like(Kh), torch.zeros_like(Vh), 0, 0)  # no HD
out_ref_causal, _ = ref_sdpa(Q, K, V, causal=True)
all_ok &= check(
    "non-answer tokens unchanged from Pass 1",
    out_two_pass[:, :ANS_START],
    out_ref_causal[:, :ANS_START],
    atol=2e-5,
)

# ─────────────────────────────────────────────────────────────────────────────
# T4  Backward: gradients w.r.t. Q, K, V, Kh, Vh
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("T4  Backward  (gradient consistency vs ground-truth joint attn)")
print("=" * 62)

def two_pass_output(Q, K, V, Kh, Vh):
    out1, lse1 = _dat_attn_with_lse(Q, K, V, causal=True)
    out_parts = []
    for b in range(B):
        q_ans  = Q[b:b+1, :, ANS_START:ANS_END, :]
        out2_b, lse2_b = _dat_attn_with_lse(q_ans, Kh[b:b+1], Vh[b:b+1], causal=False)
        merged = _merge_two_pass_lse(
            out1[b:b+1], lse1[b:b+1],
            out2_b, lse2_b,
            ANS_START, ANS_END,
        )
        out_parts.append(merged)
    return torch.cat(out_parts, dim=0)


def joint_output(Q, K, V, Kh, Vh):
    return ground_truth_joint(Q, K, V, Kh, Vh, ANS_START, ANS_END)


torch.manual_seed(2)
Q  = torch.randn(B, H, Nq, D, device=device, dtype=dtype, requires_grad=True)
K  = torch.randn(B, H, Nq, D, device=device, dtype=dtype, requires_grad=True)
V  = torch.randn(B, H, Nq, D, device=device, dtype=dtype, requires_grad=True)
Kh = torch.randn(B, H, Ns, D, device=device, dtype=dtype, requires_grad=True)
Vh = torch.randn(B, H, Ns, D, device=device, dtype=dtype, requires_grad=True)

grad_out = torch.randn(B, Nq, H, D, device=device, dtype=dtype)

# --- two-pass gradients ---
out_tp = two_pass_output(Q, K, V, Kh, Vh)
out_tp.backward(grad_out)
gQ_tp, gK_tp, gV_tp, gKh_tp, gVh_tp = Q.grad.clone(), K.grad.clone(), V.grad.clone(), Kh.grad.clone(), Vh.grad.clone()

# --- ground truth gradients ---
for t in [Q, K, V, Kh, Vh]:
    t.grad = None

out_gt2 = joint_output(Q, K, V, Kh, Vh)
out_gt2.backward(grad_out)
gQ_gt, gK_gt, gV_gt, gKh_gt, gVh_gt = Q.grad.clone(), K.grad.clone(), V.grad.clone(), Kh.grad.clone(), Vh.grad.clone()

all_ok &= check("∂L/∂Q  two-pass ≈ joint GT", gQ_tp,  gQ_gt,  atol=2e-4)
all_ok &= check("∂L/∂K  two-pass ≈ joint GT", gK_tp,  gK_gt,  atol=2e-4)
all_ok &= check("∂L/∂V  two-pass ≈ joint GT", gV_tp,  gV_gt,  atol=2e-4)
all_ok &= check("∂L/∂Kh two-pass ≈ joint GT", gKh_tp, gKh_gt, atol=2e-4)
all_ok &= check("∂L/∂Vh two-pass ≈ joint GT", gVh_tp, gVh_gt, atol=2e-4)

# ─────────────────────────────────────────────────────────────────────────────
# T5  Gradient Checkpointing compatibility
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("T5  Gradient Checkpointing (torch.utils.checkpoint)")
print("=" * 62)

torch.manual_seed(3)
Q  = torch.randn(B, H, Nq, D, device=device, dtype=dtype, requires_grad=True)
K  = torch.randn(B, H, Nq, D, device=device, dtype=dtype, requires_grad=True)
V  = torch.randn(B, H, Nq, D, device=device, dtype=dtype, requires_grad=True)
Kh = torch.randn(B, H, Ns, D, device=device, dtype=dtype, requires_grad=True)
Vh = torch.randn(B, H, Ns, D, device=device, dtype=dtype, requires_grad=True)
grad_out = torch.randn(B, Nq, H, D, device=device, dtype=dtype)

# Wrap the entire two-pass function in gradient checkpointing
out_gc = cp.checkpoint(two_pass_output, Q, K, V, Kh, Vh, use_reentrant=False)
out_gc.backward(grad_out)

gQ_gc, gK_gc, gV_gc = Q.grad.clone(), K.grad.clone(), V.grad.clone()

# Compare GC backward with regular backward
for t in [Q, K, V, Kh, Vh]:
    t.grad = None
out_reg = two_pass_output(Q, K, V, Kh, Vh)
out_reg.backward(grad_out)
gQ_reg, gK_reg, gV_reg = Q.grad.clone(), K.grad.clone(), V.grad.clone()

all_ok &= check("GC forward  matches regular", out_gc,  out_reg, atol=1e-6)
all_ok &= check("GC ∂L/∂Q   matches regular", gQ_gc,   gQ_reg,  atol=1e-5)
all_ok &= check("GC ∂L/∂K   matches regular", gK_gc,   gK_reg,  atol=1e-5)
all_ok &= check("GC ∂L/∂V   matches regular", gV_gc,   gV_reg,  atol=1e-5)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 62)
if all_ok:
    print("ALL TESTS PASSED ✓")
else:
    print("SOME TESTS FAILED ✗  — see details above")
print("=" * 62)
sys.exit(0 if all_ok else 1)
