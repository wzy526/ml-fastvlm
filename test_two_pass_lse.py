"""test_two_pass_lse.py

验证双路 attention + LSE 合并 (Two-pass + LSE merge) 的正确性。

核心数学：
    Pass 1: o₁, ℓ₁ = CausalAttn(Q, K, V)
    Pass 2: o₂, ℓ₂ = CrossAttn(Q_ans, K_hd, V_hd)  (causal=False)
    Merge:  ℓ  = logaddexp(ℓ₁, ℓ₂)
            o* = exp(ℓ₁ − ℓ) · o₁ + exp(ℓ₂ − ℓ) · o₂

等价于对 S₁ ∪ S₂ 的 joint attention，验证方法：
    直接构建 [K; K_hd] 并施加正确的 mask，与两路合并结果比较。

Run:
    python test_two_pass_lse.py
"""

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32

B   = 2    # batch size
H   = 4    # Q heads (GQA ratio = 1 for simplicity)
D   = 16   # head dim
Nq  = 20   # sequence length
Ns  = 9    # HD tokens per answer segment  (3×3)

# intention_idx: HD visible to Q[i] for i > intention_idx
INTENTION = [7, 8]   # per batch element
ANS_START = [10, 12] # per batch element  (answer range start)
ANS_END   = [16, 18] # per batch element  (answer range end, training mode)

torch.manual_seed(42)
Q    = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
K    = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
V    = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
K_hd = torch.randn(B, H, Ns, D, device=device, dtype=dtype)
V_hd = torch.randn(B, H, Ns, D, device=device, dtype=dtype)

print(f"Device  : {device}")
print(f"PyTorch : {torch.__version__}")
print(f"Shapes  : Q={tuple(Q.shape)}  K_hd={tuple(K_hd.shape)}")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute attention with optional causal mask, returning (out, lse)
# ──────────────────────────────────────────────────────────────────────────────
def sdpa_with_lse(q, k, v, mask_float=None, causal=False):
    """q,k,v: [B,H,Nq,D].  Returns out [B,H,Nq,D], lse [B,H,Nq]."""
    scale = D ** -0.5
    scores = (q @ k.transpose(-1, -2)) * scale   # [B,H,Nq,Nk]
    if causal:
        cm = torch.triu(torch.ones(q.size(2), k.size(2), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(cm, float('-inf'))
    if mask_float is not None:
        scores = scores + mask_float
    lse = torch.logsumexp(scores, dim=-1)         # [B,H,Nq]
    attn = torch.softmax(scores, dim=-1)
    out  = attn @ v                               # [B,H,Nq,D]
    return out, lse


# ──────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH: joint attention with explicit mask
#   K_joint = [K | K_hd]  (concatenated along KV dimension)
#   mask: Q[i] can attend to K_hd[m] iff i > intention_idx
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[1] GROUND TRUTH — joint attention with explicit mask")
print("=" * 60)

# Build per-batch joint mask [B, 1, Nq, Nq+Ns]
out_joint_list = []
for b in range(B):
    intn = INTENTION[b]
    K_joint = torch.cat([K[b:b+1], K_hd[b:b+1]], dim=2)    # [1, H, Nq+Ns, D]
    V_joint = torch.cat([V[b:b+1], V_hd[b:b+1]], dim=2)

    # Causal mask for K part; HD mask: Q[i] sees K_hd iff i > intn
    q_idx = torch.arange(Nq, device=device)
    k_idx = torch.arange(Nq, device=device)

    causal_col = (q_idx.unsqueeze(1) >= k_idx.unsqueeze(0))    # [Nq, Nq]
    hd_col     = (q_idx > intn).unsqueeze(1).expand(Nq, Ns)    # [Nq, Ns]
    bool_mask  = torch.cat([causal_col, hd_col], dim=1)        # [Nq, Nq+Ns]
    float_mask = torch.zeros(1, 1, Nq, Nq + Ns, device=device, dtype=dtype)
    float_mask[~bool_mask.unsqueeze(0).unsqueeze(0)] = float('-inf')

    out_b, _ = sdpa_with_lse(Q[b:b+1], K_joint, V_joint, mask_float=float_mask)
    out_joint_list.append(out_b)

out_joint = torch.cat(out_joint_list, dim=0)   # [B, H, Nq, D]
print(f"  Output shape : {tuple(out_joint.shape)}")
print(f"  Output mean  : {out_joint.mean():.6f}")


# ──────────────────────────────────────────────────────────────────────────────
# TWO-PASS + LSE MERGE
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[2] TWO-PASS + LSE MERGE")
print("=" * 60)

# Pass 1: causal attention on full sequence → out1 [B,Nq,H,D], lse1 [B,H,Nq]
out1_bhnd, lse1 = sdpa_with_lse(Q, K, V, causal=True)
out1 = out1_bhnd.permute(0, 2, 1, 3)   # [B, Nq, H, D]

# Pass 2 + merge: per batch element, per answer segment
out_merged = out1.clone()
for b in range(B):
    intn      = INTENTION[b]
    ans_start = ANS_START[b]
    ans_end   = ANS_END[b]
    Nans      = ans_end - ans_start

    q_ans = Q[b:b+1, :, ans_start:ans_start+Nans, :]   # [1, H, Nans, D]
    k_h   = K_hd[b:b+1]                                # [1, H, Ns, D]
    v_h   = V_hd[b:b+1]

    # Pass 2: non-causal cross-attention
    out2_bhnd, lse2 = sdpa_with_lse(q_ans, k_h, v_h, causal=False)
    out2 = out2_bhnd.permute(0, 2, 1, 3)               # [1, Nans, H, D]

    # LSE merge
    lse1_ans = lse1[b:b+1, :, ans_start:ans_start+Nans]   # [1, H, Nans]
    lse  = torch.logaddexp(lse1_ans, lse2)
    w1   = (lse1_ans - lse).exp().permute(0, 2, 1).unsqueeze(-1)   # [1, Nans, H, 1]
    w2   = (lse2     - lse).exp().permute(0, 2, 1).unsqueeze(-1)

    merged = w1 * out_merged[b:b+1, ans_start:ans_start+Nans] + w2 * out2
    out_merged[b:b+1, ans_start:ans_start+Nans] = merged

# Convert joint back to [B, Nq, H, D] for comparison
out_joint_bnhd = out_joint.permute(0, 2, 1, 3)   # [B, Nq, H, D]

print(f"  Output shape : {tuple(out_merged.shape)}")
print(f"  Output mean  : {out_merged.mean():.6f}")


# ──────────────────────────────────────────────────────────────────────────────
# COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON (joint vs two-pass+LSE)")
print("=" * 60)

diff      = (out_joint_bnhd - out_merged).abs()
max_diff  = diff.max().item()
mean_diff = diff.mean().item()

print(f"  Max  |joint − merged| : {max_diff:.3e}")
print(f"  Mean |joint − merged| : {mean_diff:.3e}")

THRESHOLD = 1e-5
if max_diff < THRESHOLD:
    print(f"\n  ✓ OUTPUTS MATCH  (threshold {THRESHOLD})")
else:
    print(f"\n  ✗ OUTPUTS DIFFER — per-position breakdown:")
    for b in range(B):
        d_b = diff[b].mean(dim=(0, -1))  # [Nq]
        top5 = d_b.topk(5)
        print(f"    batch {b}: worst Q positions {top5.indices.tolist()}"
              f"  diff {[f'{v:.2e}' for v in top5.values.tolist()]}")

# ──────────────────────────────────────────────────────────────────────────────
# NORMALIZATION CHECK: attention weights sum to 1 after merge
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NORMALIZATION CHECK — verify ∑ α = 1 after merge")
print("=" * 60)

for b in range(B):
    intn      = INTENTION[b]
    ans_start = ANS_START[b]
    ans_end   = ANS_END[b]
    Nans      = ans_end - ans_start

    q_ans = Q[b:b+1, :, ans_start:ans_start+Nans, :]
    k_h   = K_hd[b:b+1]; v_h = V_hd[b:b+1]

    # Compute both raw attention weight arrays
    scale = D ** -0.5

    # From Pass 1 (causal): only look at the ans_start..ans_end rows
    sc1 = (q_ans @ K[b:b+1].transpose(-1,-2)) * scale   # [1,H,Nans,Nq]
    # Apply causal mask for q positions [ans_start..ans_end)
    q_idx_abs = torch.arange(ans_start, ans_start+Nans, device=device)
    k_idx_abs = torch.arange(Nq, device=device)
    causal_col = (q_idx_abs.unsqueeze(1) >= k_idx_abs.unsqueeze(0))  # [Nans, Nq]
    float_m1 = torch.zeros(1, 1, Nans, Nq, device=device, dtype=dtype)
    float_m1[~causal_col.unsqueeze(0).unsqueeze(0)] = float('-inf')
    sc1 = sc1 + float_m1

    # From Pass 2 (non-causal)
    sc2 = (q_ans @ k_h.transpose(-1,-2)) * scale         # [1,H,Nans,Ns]

    # Joint softmax denominator
    joint_scores = torch.cat([sc1, sc2], dim=-1)          # [1,H,Nans,Nq+Ns]
    joint_attn   = torch.softmax(joint_scores, dim=-1)
    sum_attn     = joint_attn.sum(dim=-1)                  # [1,H,Nans]
    max_dev = (sum_attn - 1.0).abs().max().item()
    print(f"  batch {b}: max |∑α − 1| = {max_dev:.2e}  {'✓' if max_dev < 1e-5 else '✗'}")

print("=" * 60)
