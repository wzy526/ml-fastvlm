"""test_dat_flex_attention.py

Naive DAT attention (attn_mask_4d + SDPA)
  vs
FlexAttention with mask_mod (mask := kv_idx ≤ q_upper_bounds[b, q_idx])

Key insight:  DAT's mask in extended-KV coordinate space is purely causal
(monotone upper-bound per Q). This makes it the most efficient case for
FlexAttention / FlashMask.

Run:
    conda activate fastvlm
    python test_dat_flex_attention.py
"""

import torch
import torch.nn.functional as F

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32   # float32 for exact numerical comparison

B   = 2    # batch size
H   = 4    # Q heads
Hkv = 2    # KV heads  (GQA ratio = H // Hkv = 2)
D   = 16   # head dim
Nq  = 20   # original sequence length (tokens)
Ns  = 9    # HD tokens per answer range  (3×3 grid)

GQA = H // Hkv

print(f"Device  : {device}")
print(f"PyTorch : {torch.__version__}")

# ────────────────────────────────────────────────────────────────
# FlexAttention availability
# ────────────────────────────────────────────────────────────────
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_AVAILABLE = True
    print("✓ FlexAttention available\n")
except ImportError:
    FLEX_AVAILABLE = False
    print("✗ FlexAttention NOT available (requires PyTorch ≥ 2.5)\n")

# ────────────────────────────────────────────────────────────────
# Synthetic image_range_list
#   [(lr_start, lr_end, lr_h, lr_w),
#    [ans_start, ans_end, intention_idx], ...]
# ────────────────────────────────────────────────────────────────
image_range_list = [
    [(2, 6, 2, 2),  [10, 16, 7]],   # batch 0 — intention at 7
    [(1, 5, 2, 2),  [12, 18, 8]],   # batch 1 — intention at 8
]

# ────────────────────────────────────────────────────────────────
# Random QKV  (already in multi-head layout)
# ────────────────────────────────────────────────────────────────
torch.manual_seed(42)
Q    = torch.randn(B, H,   Nq, D, device=device, dtype=dtype)
K    = torch.randn(B, Hkv, Nq, D, device=device, dtype=dtype)
V    = torch.randn(B, Hkv, Nq, D, device=device, dtype=dtype)
K_hd = torch.randn(B, Hkv, Ns, D, device=device, dtype=dtype)   # HD features, per batch
V_hd = torch.randn(B, Hkv, Ns, D, device=device, dtype=dtype)

# ────────────────────────────────────────────────────────────────
# Helper: GQA expand  [B, Hkv, S, D] → [B, H, S, D]
# ────────────────────────────────────────────────────────────────
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, Hkv, S, D = x.shape
    return (x.unsqueeze(2)
             .expand(B, Hkv, n_rep, S, D)
             .reshape(B, Hkv * n_rep, S, D))

# ────────────────────────────────────────────────────────────────
# Build extended KV + bool mask + q_upper_bounds
#   for ONE batch element  (mirrors _reorganize_kv_mask_and_pos)
#
# Returns:
#   K_ext  : [kv_len, Hkv, D]
#   V_ext  : [kv_len, Hkv, D]
#   mask   : [Nq, kv_len]  bool  — True = can attend
#   q_ub   : [Nq]          int   — extended-KV upper-bound for each Q position
#
# Insight:
#   mask[i, k]  ⟺  k ≤ q_ub[i]
#   because in the extended-KV coordinate space the entire pattern is causal.
# ────────────────────────────────────────────────────────────────
def build_one(Kb, Vb, Khd_b, Vhd_b, ranges):
    """
    Kb, Vb      : [Hkv, Nq, D]
    Khd_b,Vhd_b : [Hkv, Ns, D]
    ranges      : [(lr_start,lr_end,lr_h,lr_w), [ans_start, ans_end, intn], ...]
    """
    k_split, v_split, m_cols = [], [], []
    q_ub_parts = []          # extended-KV positions of each Q token, in original Q order
    insert_ctr = 0           # next original-KV position not yet emitted
    pos_ctr    = 0           # current position in extended-KV sequence

    answer_ranges = ranges[1:]    # first entry is LR info, skip

    for ar in answer_ranges:
        ans_start, ans_end, intn = ar   # intn = intention_idx

        # ── Part 1 : original K[insert_ctr .. intn]  (inclusive) ──────────
        sl = intn + 1 - insert_ctr
        k_split.append(Kb[:, insert_ctr:intn+1, :].permute(1, 0, 2))  # [sl, Hkv, D]
        v_split.append(Vb[:, insert_ctr:intn+1, :].permute(1, 0, 2))

        # Causal mask columns: Q[i] can attend to K[j] if i ≥ j_orig
        orig_j = torch.arange(insert_ctr, intn + 1, device=device)         # [sl]
        q_rows = torch.arange(Nq, device=device).unsqueeze(1)              # [Nq, 1]
        m_cols.append(q_rows >= orig_j)                                     # [Nq, sl]

        # Extended-KV positions for Q[insert_ctr..intn]
        q_ub_parts.append(torch.arange(pos_ctr, pos_ctr + sl, device=device))
        pos_ctr += sl

        # ── Part 2 : HD tokens  (Ns) ──────────────────────────────────────
        k_split.append(Khd_b.permute(1, 0, 2))   # [Ns, Hkv, D]
        v_split.append(Vhd_b.permute(1, 0, 2))

        # HD visible to Q[i] iff i > intn
        hd_vis = (torch.arange(Nq, device=device) > intn).unsqueeze(1).expand(Nq, Ns)
        m_cols.append(hd_vis)                                               # [Nq, Ns]
        # NOTE: no q_ub_parts entry here — HD tokens are KV-only
        pos_ctr += Ns

        # ── Part 3 : original K[intn+1 .. ans_end]  (inclusive) ───────────
        if ans_end > intn:
            sl = ans_end - intn
            k_split.append(Kb[:, intn+1:ans_end+1, :].permute(1, 0, 2))
            v_split.append(Vb[:, intn+1:ans_end+1, :].permute(1, 0, 2))

            orig_j = torch.arange(intn + 1, ans_end + 1, device=device)
            m_cols.append(q_rows >= orig_j)                                 # [Nq, sl]

            q_ub_parts.append(torch.arange(pos_ctr, pos_ctr + sl, device=device))
            pos_ctr += sl

        insert_ctr = ans_end + 1

    # ── Trailing : original K[insert_ctr .. Nq-1] ─────────────────────────
    if insert_ctr < Nq:
        sl = Nq - insert_ctr
        k_split.append(Kb[:, insert_ctr:Nq, :].permute(1, 0, 2))
        v_split.append(Vb[:, insert_ctr:Nq, :].permute(1, 0, 2))

        orig_j = torch.arange(insert_ctr, Nq, device=device)
        m_cols.append(q_rows >= orig_j)                                     # [Nq, sl]

        q_ub_parts.append(torch.arange(pos_ctr, pos_ctr + sl, device=device))
        pos_ctr += sl

    K_ext = torch.cat(k_split,    dim=0)   # [kv_len, Hkv, D]
    V_ext = torch.cat(v_split,    dim=0)
    mask  = torch.cat(m_cols,     dim=1)   # [Nq, kv_len]  bool
    q_ub  = torch.cat(q_ub_parts, dim=0)   # [Nq]          int

    return K_ext, V_ext, mask, q_ub


# ── Build for every batch element ─────────────────────────────────────────
k_list, v_list, m_list, ub_list = [], [], [], []
for b in range(B):
    ke, ve, mb, ub = build_one(K[b], V[b], K_hd[b], V_hd[b], image_range_list[b])
    k_list.append(ke); v_list.append(ve)
    m_list.append(mb); ub_list.append(ub)

# ── Pad to uniform kv_len ─────────────────────────────────────────────────
kv_len = max(k.shape[0] for k in k_list)
print(f"Shapes — Q: {Q.shape}  kv_len: {kv_len}  (Nq={Nq} + Ns={Ns} = {Nq+Ns})")

def pad_kv(t, n):
    d = n - t.shape[0]
    return F.pad(t, (0,0, 0,0, 0,d)) if d > 0 else t

def pad_mask(m, n):
    d = n - m.shape[1]
    return F.pad(m, (0, d), value=False) if d > 0 else m

K_ext_batch = torch.stack([pad_kv(k, kv_len) for k in k_list])      # [B, kv_len, Hkv, D]
V_ext_batch = torch.stack([pad_kv(v, kv_len) for v in v_list])
M_bool       = torch.stack([pad_mask(m, kv_len) for m in m_list])    # [B, Nq, kv_len]
q_upper      = torch.stack(ub_list)                                   # [B, Nq]

# Reshape + GQA expand: [B, H, kv_len, D]
K_full = repeat_kv(K_ext_batch.permute(0, 2, 1, 3), GQA)
V_full = repeat_kv(V_ext_batch.permute(0, 2, 1, 3), GQA)

# ════════════════════════════════════════════════════════════════
# SANITY CHECK — mask == (k ≤ q_upper) pointwise
#   This must pass before we trust either implementation.
# ════════════════════════════════════════════════════════════════
print("\n[Sanity] Verifying  mask[i,k]  ⟺  k ≤ q_upper[b,i] ...")
k_range = torch.arange(kv_len, device=device)
M_ref   = k_range.unsqueeze(0).unsqueeze(0) <= q_upper.unsqueeze(2)   # [B, Nq, kv_len]
mismatch = (M_bool != M_ref).sum().item()
if mismatch == 0:
    print("  ✓ Equivalence confirmed — the two representations are identical\n")
else:
    bad = (M_bool != M_ref).nonzero()
    print(f"  ✗ {mismatch} mismatches at positions: {bad[:8].tolist()}\n")

# ════════════════════════════════════════════════════════════════
# 1. NAIVE IMPLEMENTATION
#    Build float -inf mask [B, 1, Nq, kv_len] → F.scaled_dot_product_attention
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("[1] NAIVE  (attn_mask_4d + SDPA)")
print("=" * 60)

attn_mask_float = torch.zeros(B, 1, Nq, kv_len, device=device, dtype=dtype)
attn_mask_float = attn_mask_float.masked_fill(~M_bool.unsqueeze(1), float('-inf'))
print(f"  attn_mask_4d shape : {attn_mask_float.shape}  "
      f"({attn_mask_float.numel() * attn_mask_float.element_size() / 1024:.1f} KB)")

with torch.no_grad():
    out_naive = F.scaled_dot_product_attention(
        Q, K_full, V_full,
        attn_mask=attn_mask_float,
        is_causal=False,
    )   # [B, H, Nq, D]

print(f"  Output shape : {out_naive.shape}")
print(f"  Output mean  : {out_naive.mean():.6f}")
print(f"  Output norm  : {out_naive.norm():.4f}")

# ════════════════════════════════════════════════════════════════
# 2. FLEX ATTENTION  (Path 1 — mask_mod + block_mask)
#    mask_mod: kv_idx ≤ q_upper[b, q_idx]
# ════════════════════════════════════════════════════════════════
if FLEX_AVAILABLE:
    print("\n" + "=" * 60)
    print("[2] FLEX ATTENTION  (create_block_mask + flex_attention)")
    print("=" * 60)

    # q_upper is captured as closure: [B, Nq] int tensor
    # mask_mod(b, h, q_idx, kv_idx) → bool
    def dat_mask_mod(b, h, q_idx, kv_idx):
        return kv_idx <= q_upper[b, q_idx]

    block_mask = create_block_mask(
        dat_mask_mod,
        B=B,
        H=None,          # same mask for all heads
        Q_LEN=Nq,
        KV_LEN=kv_len,
        device=device,
    )
    print(f"  BlockMask : {block_mask}")

    with torch.no_grad():
        out_flex = flex_attention(
            Q, K_full, V_full,
            block_mask=block_mask,
        )   # [B, H, Nq, D]

    print(f"  Output shape : {out_flex.shape}")
    print(f"  Output mean  : {out_flex.mean():.6f}")
    print(f"  Output norm  : {out_flex.norm():.4f}")

    # ──────────────────────────────────────────────────────────
    # Comparison
    # ──────────────────────────────────────────────────────────
    diff      = (out_naive - out_flex).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Max  |naive - flex| : {max_diff:.3e}")
    print(f"  Mean |naive - flex| : {mean_diff:.3e}")

    THRESHOLD = 1e-4
    if max_diff < THRESHOLD:
        print(f"\n  ✓ OUTPUTS MATCH  (threshold {THRESHOLD})")
    else:
        print(f"\n  ✗ OUTPUTS DIFFER  — top-5 worst Q positions:")
        worst = diff.mean(dim=-1).mean(dim=1)   # [B, Nq]
        for b in range(B):
            top5 = worst[b].topk(5)
            print(f"    batch {b}: Q pos {top5.indices.tolist()}  diff {top5.values.tolist()}")
    print("=" * 60)

else:
    print("\nFlexAttention not available — only naive output produced.")
    print("Install PyTorch ≥ 2.5 to run the comparison.")

# ════════════════════════════════════════════════════════════════
# 3. MANUAL SPOT CHECK
#    Print q_upper_bounds and verify a few key positions
# ════════════════════════════════════════════════════════════════
print("\n[Spot check]  q_upper_bounds per batch element")
for b in range(B):
    intn = image_range_list[b][1][2]
    ub   = q_upper[b].tolist()
    print(f"  batch {b}  intention_idx={intn}")
    print(f"    q_upper = {ub}")
    print(f"    Q[{intn  }] upper={ub[intn  ]}  → HD starts at {intn+1}:  "
          f"{'CAN see HD' if ub[intn  ] >= intn+1 else 'CANNOT see HD'}")
    print(f"    Q[{intn+1}] upper={ub[intn+1]}  → HD starts at {intn+1}:  "
          f"{'CAN see HD' if ub[intn+1] >= intn+1 else 'CANNOT see HD'}")
