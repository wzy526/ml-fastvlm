"""Quantitative diagnostic of attention propagation in DAT.

Three measurements per sample, aggregated across N HR-Bench samples.

(A) hd_attention_share[L]
    At each DAT layer's LSE merge, for each answer token:
        w_HD = exp(lse2_gated - lse_total)
    Direct measurement of how much of the merged attention weight is
    placed on the HD path at this layer.

(B) hd_output_contribution[L]
    Numerical perturbation HD induces on the merged output:
        ||w_HD * out2|| / ||w_LR*out1 + w_HD*out2||
    A high w_HD with a low contribution means "model gives HD attention
    but HD has no useful value". A low w_HD with low contribution means
    "model is shutting HD out". Together with (A), separates attention
    suppression from value collapse.

(C) cross_layer_q_alignment[L -> L_next]
    Zhuofan's "this layer's Q as previous layer's K — how much attn score
    does it aggregate?" diagnostic:
        attn = softmax(Q_L[ans] @ Q_{L_next}[all]^T / sqrt(d_head))
    Per-head, then averaged. For each answer token, we report how the
    attention mass distributes over sequence regions (system, LR image,
    QA prefix, answer self). High mass on LR image position i in
    alignment of pair (L, L_next) means "layer L_next's Q at the answer
    aligns with layer L's Q at LR token i" — i.e. cross-layer consistency
    of the LR-image query subspace.

Output:
    - Console summary tables
    - JSON dump of per-sample / per-layer stats

Usage:
    python scripts/qwen2_5vl_adl_0519/_diagnose_attention_propagation.py \
        --ckpt /root/autodl-tmp/vldat_experiments/<some-merged-ckpt> \
        --n_samples 12 \
        --out_dir /root/autodl-tmp/diag_attn_prop_<tag>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/root/autodl-tmp/lmms-eval")

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from llava.model.language_model.modeling_qwen2_5vl_dat import (  # noqa: E402
    Qwen2_5_VLAttentionDAT,
)


# ============================================================================
# Capture
# ============================================================================
class PropagationCapture:
    """Captures, for each forward pass:

    * Q states from every transformer layer's q_proj (hook).
    * LSE merge stats from every DAT layer (monkey-patched _merge_two_pass_lse).
    * image_range_list (lr_range + answer ranges) from the first DAT layer
      (pre-forward hook).
    """

    def __init__(self):
        # layer_idx -> q_proj output tensor (cpu, fp32, [B, Nq, C])
        self.q_states: dict[int, torch.Tensor] = {}
        # one record per (sample, layer, answer-range) — list of dicts
        self.merge_records: list[dict] = []
        # captured once per forward
        self.image_range_list: list | None = None
        # current sample index, set externally between samples
        self.current_sample_idx: int = 0
        # n_heads / head_dim filled in install()
        self.n_heads: int = 0
        self.head_dim: int = 0

    def reset_sample(self):
        self.q_states.clear()
        self.image_range_list = None

    # ----- install hooks / patches ---------------------------------------
    def install(self, model):
        q_handles = []
        pre_handles = []  # kept for API compat; no pre-hooks used now
        all_layer_indices: list[int] = []
        dat_layer_indices: list[int] = []

        first_dat_attn = None
        for name, mod in model.named_modules():
            if not name.endswith(".self_attn"):
                continue
            try:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
            except Exception:
                continue

            if not hasattr(mod, "q_proj"):
                continue

            q_proj = mod.q_proj
            q_proj._diag_layer_idx = layer_idx
            h = q_proj.register_forward_hook(self._q_hook)
            q_handles.append(h)
            all_layer_indices.append(layer_idx)

            if isinstance(mod, Qwen2_5_VLAttentionDAT):
                dat_layer_indices.append(layer_idx)
                self._patch_merge(mod, layer_idx)
                # Patch _generate_offsets_and_sample to capture image_range_list
                # (forward_pre_hook with_kwargs=True was unreliable because the
                # decoder layer may pass image_range_list positionally).
                self._patch_meta(mod)
                if first_dat_attn is None:
                    first_dat_attn = mod
                    if self.n_heads == 0:
                        self.n_heads = int(mod.num_heads)
                        self.head_dim = int(mod.head_dim)

        if first_dat_attn is None:
            raise RuntimeError("No DAT attention layer found in model")

        return q_handles, pre_handles, sorted(set(all_layer_indices)), sorted(dat_layer_indices)

    # ----- hook bodies ---------------------------------------------------
    def _q_hook(self, module, inputs, output):
        layer_idx = getattr(module, "_diag_layer_idx", id(module))
        # output: [B, Nq, hidden_size]; store CPU fp32 to keep GPU light
        self.q_states[layer_idx] = output.detach().float().cpu()

    def _patch_meta(self, attn_mod):
        """Patch `_generate_offsets_and_sample` to capture image_range_list.

        That method receives image_range_list positionally, so capture is
        robust regardless of how the caller (DecoderLayer.forward) routes it.
        A forward_pre_hook with_kwargs=True turned out to be unreliable here
        because the DecoderLayer routes DAT kwargs positionally.
        """
        orig = attn_mod._generate_offsets_and_sample
        cap = self

        def patched(query_states, image_hd_features, image_range_list,
                    b_idx, hd_feat_idx):
            if cap.image_range_list is None and image_range_list is not None:
                cap.image_range_list = image_range_list
            return orig(query_states, image_hd_features, image_range_list,
                        b_idx, hd_feat_idx)

        attn_mod._generate_offsets_and_sample = patched

    def _patch_merge(self, attn_mod, layer_idx: int):
        orig = attn_mod._merge_two_pass_lse
        cap = self

        def patched(out1, lse1, out2, lse2, ans_start, ans_end):
            with torch.no_grad():
                lse1_ans = lse1[:, :, ans_start:ans_end].float()
                lse2_f = lse2.float()
                out1_ans = out1[:, ans_start:ans_end, :, :]

                if attn_mod.hd_gate is not None:
                    gate_log = F.logsigmoid(attn_mod.hd_gate.detach()).float()
                    hd_gate_val = float(attn_mod.hd_gate.detach().sigmoid().item())
                else:
                    gate_log = torch.zeros((), device=lse2_f.device, dtype=lse2_f.dtype)
                    hd_gate_val = 1.0

                lse2_gated = lse2_f + gate_log
                lse_total = torch.logaddexp(lse1_ans, lse2_gated)
                w1 = (lse1_ans - lse_total).exp()  # [B, H, Nans]
                w2 = (lse2_gated - lse_total).exp()

                w_HD_pt = w2.mean(dim=1)[0].cpu().numpy()  # [Nans]
                w_LR_pt = w1.mean(dim=1)[0].cpu().numpy()

                w1_b = w1.permute(0, 2, 1).unsqueeze(-1)  # [B, Nans, H, 1]
                w2_b = w2.permute(0, 2, 1).unsqueeze(-1)
                contrib_lr = (w1_b * out1_ans).float()
                contrib_hd = (w2_b * out2).float()
                merged = contrib_lr + contrib_hd

                n_lr = contrib_lr.reshape(1, contrib_lr.size(1), -1).norm(dim=-1)[0]
                n_hd = contrib_hd.reshape(1, contrib_hd.size(1), -1).norm(dim=-1)[0]
                n_merge = merged.reshape(1, merged.size(1), -1).norm(dim=-1)[0]
                ratio = (n_hd / n_merge.clamp_min(1e-8)).cpu().numpy()

                cap.merge_records.append({
                    "sample_idx": cap.current_sample_idx,
                    "layer_idx": layer_idx,
                    "ans_start": int(ans_start),
                    "ans_end": int(ans_end),
                    "n_ans_tokens": int(ans_end - ans_start),
                    "hd_gate_sigmoid": hd_gate_val,
                    "w_HD_mean": float(w_HD_pt.mean()),
                    "w_HD_std": float(w_HD_pt.std()),
                    "w_HD_min": float(w_HD_pt.min()),
                    "w_HD_max": float(w_HD_pt.max()),
                    "w_LR_mean": float(w_LR_pt.mean()),
                    "norm_hd_mean": float(n_hd.cpu().numpy().mean()),
                    "norm_lr_mean": float(n_lr.cpu().numpy().mean()),
                    "norm_merged_mean": float(n_merge.cpu().numpy().mean()),
                    "hd_to_merged_ratio_mean": float(ratio.mean()),
                    "hd_to_merged_ratio_max": float(ratio.max()),
                })

            return orig(out1, lse1, out2, lse2, ans_start, ans_end)

        attn_mod._merge_two_pass_lse = patched


# ============================================================================
# Cross-layer Q-alignment (Zhuofan's propagation diagnostic)
# ============================================================================
def compute_q_alignment(
    capturer: PropagationCapture,
    all_layer_indices: list[int],
    sample_idx: int,
) -> list[dict]:
    """For each adjacent (L, L_next) pair, softmax(Q_L[ans] @ Q_{L_next}[all]^T)
    per head, averaged across heads. Bucket the attention mass into sequence
    regions and report.
    """
    if capturer.image_range_list is None or capturer.n_heads == 0:
        return []

    irl_b0 = capturer.image_range_list[0]
    if len(irl_b0) < 2:
        return []
    lr_start, lr_end, lr_h, lr_w = irl_b0[0]
    ans_start_0, ans_end_0, intention_0 = irl_b0[1]

    n_heads = capturer.n_heads
    head_dim = capturer.head_dim
    inv_sqrt = 1.0 / float(head_dim) ** 0.5

    pair_stats = []
    sorted_layers = sorted(all_layer_indices)

    for i in range(len(sorted_layers) - 1):
        L = sorted_layers[i]
        L_next = sorted_layers[i + 1]
        Q_L = capturer.q_states.get(L, None)
        Q_next = capturer.q_states.get(L_next, None)
        if Q_L is None or Q_next is None:
            continue
        if Q_L.shape != Q_next.shape:
            continue

        B, Nq, C = Q_L.shape
        if C != n_heads * head_dim:
            continue
        # Resolve answer span the SAME way Qwen2_5_VLAttentionDAT.forward does:
        #   ans_end > 0  → q span = [ans_start, ans_end)
        #   ans_end <= 0 → generation mode: q span = [intention+1, Nq)
        if ans_end_0 > 0:
            qa_start = ans_start_0
            qa_end = ans_end_0
        else:
            qa_start = intention_0 + 1
            qa_end = Nq
        qa_end = min(qa_end, Nq)
        if qa_end <= qa_start or qa_start >= Nq:
            continue

        # Reshape to per-head: [n_heads, Nq, head_dim]
        Q_L_h = Q_L[0].view(Nq, n_heads, head_dim).transpose(0, 1).contiguous()
        Q_next_h = Q_next[0].view(Nq, n_heads, head_dim).transpose(0, 1).contiguous()

        q_ans = Q_L_h[:, qa_start:qa_end, :]  # [H, Nans, hd]
        # scores: [H, Nans, Nq]
        scores = torch.matmul(q_ans, Q_next_h.transpose(-1, -2)) * inv_sqrt
        attn = torch.softmax(scores, dim=-1)
        attn_avg = attn.mean(dim=0).numpy()  # [Nans, Nq]

        # Bucket boundaries
        buckets = {
            "system": (0, max(0, lr_start)),
            "lr_image": (lr_start, lr_end),
            "qa_prefix": (lr_end, qa_start),
            "answer_self": (qa_start, qa_end),
        }
        mass = {}
        for name, (lo, hi) in buckets.items():
            if lo < hi and lo < Nq:
                hi_c = min(hi, Nq)
                mass[name] = float(attn_avg[:, lo:hi_c].sum(axis=1).mean())
            else:
                mass[name] = 0.0

        # Diagonal mass: how much does each answer-token attend to itself
        # at the next layer? attn_avg[i, qa_start + i]
        diag_vals = []
        for r in range(attn_avg.shape[0]):
            col = qa_start + r
            if 0 <= col < attn_avg.shape[1]:
                diag_vals.append(float(attn_avg[r, col]))
        diag_mean = float(np.mean(diag_vals)) if diag_vals else 0.0

        pair_stats.append({
            "sample_idx": sample_idx,
            "L": int(L),
            "L_next": int(L_next),
            "n_ans_tokens": int(qa_end - qa_start),
            "mass_system": mass["system"],
            "mass_lr_image": mass["lr_image"],
            "mass_qa_prefix": mass["qa_prefix"],
            "mass_answer_self": mass["answer_self"],
            "diag_self_alignment": diag_mean,
        })

    return pair_stats


# ============================================================================
# Aggregation
# ============================================================================
def aggregate_merge_records(merge_records, dat_layer_indices):
    """Group by layer_idx, average per-sample metrics across samples."""
    by_layer = {L: [] for L in dat_layer_indices}
    for r in merge_records:
        by_layer.setdefault(r["layer_idx"], []).append(r)

    summary = {}
    for L in dat_layer_indices:
        recs = by_layer.get(L, [])
        if not recs:
            continue
        keys = (
            "hd_gate_sigmoid",
            "w_HD_mean", "w_HD_std", "w_HD_min", "w_HD_max", "w_LR_mean",
            "norm_hd_mean", "norm_lr_mean", "norm_merged_mean",
            "hd_to_merged_ratio_mean", "hd_to_merged_ratio_max",
        )
        agg = {}
        for k in keys:
            vals = [r[k] for r in recs if r.get(k) is not None]
            if vals:
                agg[f"{k}__mean"] = float(np.mean(vals))
                agg[f"{k}__std"] = float(np.std(vals))
        agg["n_samples"] = len(recs)
        summary[L] = agg
    return summary


def aggregate_q_alignment(pair_stats_all):
    """Group by (L, L_next), average across samples."""
    by_pair: dict[tuple[int, int], list] = {}
    for s in pair_stats_all:
        key = (s["L"], s["L_next"])
        by_pair.setdefault(key, []).append(s)
    summary = {}
    for (L, L_next), records in by_pair.items():
        keys = ("mass_system", "mass_lr_image", "mass_qa_prefix",
                "mass_answer_self", "diag_self_alignment")
        agg = {}
        for k in keys:
            vals = [r[k] for r in records]
            agg[f"{k}__mean"] = float(np.mean(vals))
            agg[f"{k}__std"] = float(np.std(vals))
        agg["n_samples"] = len(records)
        summary[f"L{L}_to_L{L_next}"] = agg
    return summary


# ============================================================================
# Print helpers
# ============================================================================
def print_merge_summary(summary, dat_layer_indices):
    print("\n" + "=" * 100)
    print("(A) HD attention share + (B) HD output contribution per DAT layer")
    print("=" * 100)
    hdr = "{:>6s} | {:>10s} | {:>14s} | {:>14s} | {:>14s} | {:>14s} | {:>14s}"
    print(hdr.format(
        "layer", "σ(hd_gate)", "w_HD_mean", "w_HD_max",
        "norm_hd/merge", "norm_lr/merge_proxy", "ratio_max"))
    print("-" * 100)
    for L in dat_layer_indices:
        s = summary.get(L)
        if s is None:
            continue
        norm_proxy_lr = 1.0 - s["hd_to_merged_ratio_mean__mean"]
        print(hdr.format(
            f"L{L}",
            f"{s['hd_gate_sigmoid__mean']:.4f}",
            f"{s['w_HD_mean__mean']:.4f}",
            f"{s['w_HD_max__mean']:.4f}",
            f"{s['hd_to_merged_ratio_mean__mean']:.4f}",
            f"{norm_proxy_lr:.4f}",
            f"{s['hd_to_merged_ratio_max__mean']:.4f}",
        ))
    print("\nInterpretation:")
    print("  w_HD_mean ~ σ(hd_gate)/(σ(hd_gate)+1) baseline if lse1≈lse2.")
    print("    w_HD ≫ baseline => HD attention is meaningful at this layer.")
    print("    w_HD ≪ baseline => model actively avoids HD here.")
    print("  norm_hd/merge ≈ 0 with w_HD > 0 => HD path is attended but its V is "
          "near-zero (value collapse).")


def print_qalign_summary(qalign_summary):
    if not qalign_summary:
        print("\n(C) Cross-layer Q alignment: SKIPPED (no q_states captured).")
        return
    print("\n" + "=" * 100)
    print("(C) Cross-layer Q alignment (Zhuofan propagation): "
          "softmax(Q_L[ans]@Q_next[all]^T)")
    print("=" * 100)
    hdr = "{:>16s} | {:>8s} | {:>10s} | {:>10s} | {:>10s} | {:>14s}"
    print(hdr.format(
        "L → L_next", "system", "lr_image", "qa_pref", "ans_self", "diag_self"))
    print("-" * 100)
    for key, s in sorted(qalign_summary.items(),
                         key=lambda kv: int(kv[0].split("_")[0][1:])):
        print(hdr.format(
            key,
            f"{s['mass_system__mean']:.3f}",
            f"{s['mass_lr_image__mean']:.3f}",
            f"{s['mass_qa_prefix__mean']:.3f}",
            f"{s['mass_answer_self__mean']:.3f}",
            f"{s['diag_self_alignment__mean']:.4f}",
        ))
    print("\nInterpretation:")
    print("  mass_lr_image grows with depth => answer queries progressively pull")
    print("    their attention into LR-image positions of preceding layers")
    print("    (HD's effect must propagate THROUGH the LR-image queries).")
    print("  mass_answer_self high & growing => layers are tightening on the")
    print("    answer subspace rather than reaching back to image.")
    print("  diag_self_alignment ~ 1 => same answer token aligns to itself")
    print("    across layers (no representational drift).")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/root/autodl-tmp/vldat_experiments/0514_full_1d5l_sa1b_ivcap_zeroinit-merged",
        help="Path to a merged DAT checkpoint.",
    )
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument(
        "--out_dir",
        default="/root/autodl-tmp/diag_attn_prop_zeroinit",
    )
    parser.add_argument("--device", default="cuda:0",
                        help="Primary compute device for small ops (e.g. cuda:0).")
    parser.add_argument("--device_map", default="auto",
                        help="HF device_map for from_pretrained. Use 'auto' to "
                             "spread across all free GPUs (recommended when training "
                             "is occupying one or more cards). Use 'cuda:N' to pin "
                             "to a single GPU.")
    parser.add_argument("--no_qalign", action="store_true",
                        help="Skip the cross-layer Q-alignment matrix to save memory.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] instantiating Qwen2_5_DATVL adapter from {args.ckpt}")
    from lmms_eval.models.simple.qwen2_5_dat_vl import Qwen2_5_DATVL
    adapter = Qwen2_5_DATVL(
        pretrained=args.ckpt,
        device=args.device,
        device_map=args.device_map,
        batch_size=1,
        attn_implementation="sdpa",
    )

    print("[2/5] installing q_proj hooks + patching _merge_two_pass_lse")
    capturer = PropagationCapture()
    q_handles, pre_handles, all_layer_indices, dat_layer_indices = capturer.install(adapter._model)
    print(f"  q_proj hooks on {len(all_layer_indices)} layers")
    print(f"  DAT layers patched: {dat_layer_indices}")
    print(f"  n_heads={capturer.n_heads}, head_dim={capturer.head_dim}")

    print("[3/5] loading HR-Bench 4K samples")
    from datasets import load_dataset
    ds = load_dataset(
        "DreamMr/HR-Bench",
        "hrbench_version_split",
        split="hrbench_4k",
        cache_dir="/root/autodl-tmp/cache/huggingface/datasets",
    )
    chosen = list(range(min(args.n_samples, len(ds))))
    print(f"  selected {len(chosen)} samples")

    print("[4/5] running generation + capturing propagation stats")
    from lmms_eval.api.instance import Instance
    from lmms_eval.tasks.hrbench.utils import (
        decode_base64_to_image, hrbench_doc_to_text, hrbench_doc_to_visual,
    )

    TASK = "hrbench4k"
    SPLIT = "hrbench_4k"

    class _DocLookup:
        def __init__(self, ds_obj):
            self._ds = ds_obj
        def __getitem__(self, idx):
            return self._ds[int(idx)]

    adapter.task_dict = {TASK: {SPLIT: _DocLookup(ds)}}

    pair_stats_all: list[dict] = []
    sample_meta: list[dict] = []

    for i, idx in enumerate(chosen):
        doc = ds[int(idx)]
        try:
            _ = decode_base64_to_image(doc["image"])
        except Exception as e:
            print(f"  [WARN] sample {idx} image decode failed: {e}")
            continue

        prompt_text = hrbench_doc_to_text(doc)
        instance = Instance(
            request_type="generate_until",
            arguments=(
                prompt_text,
                {"max_new_tokens": 1, "temperature": 0.0},
                hrbench_doc_to_visual,
                int(idx),
                TASK,
                SPLIT,
            ),
            idx=int(idx),
            metadata={"task": TASK, "doc_id": int(idx), "repeats": 0},
        )
        instance.doc = doc

        capturer.current_sample_idx = i
        capturer.reset_sample()
        try:
            with torch.no_grad():
                _ = adapter.generate_until([instance])
        except Exception as e:
            print(f"  [WARN] sample {idx} forward failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

        if not args.no_qalign and capturer.q_states:
            ps = compute_q_alignment(capturer, all_layer_indices, i)
            pair_stats_all.extend(ps)

        sample_meta.append({
            "sample_idx": i,
            "ds_idx": int(idx),
            "category": str(doc.get("category", "?")),
            "question": doc["question"][:80],
        })

        if (i + 1) % 4 == 0:
            print(f"  processed {i + 1}/{len(chosen)}")

    print("[5/5] aggregating + writing outputs")
    merge_summary = aggregate_merge_records(capturer.merge_records, dat_layer_indices)
    qalign_summary = aggregate_q_alignment(pair_stats_all) if pair_stats_all else {}

    print_merge_summary(merge_summary, dat_layer_indices)
    print_qalign_summary(qalign_summary)

    json_out = {
        "ckpt": args.ckpt,
        "n_samples": len(sample_meta),
        "samples": sample_meta,
        "dat_layer_indices": dat_layer_indices,
        "merge_summary": {f"L{k}": v for k, v in merge_summary.items()},
        "qalign_summary": qalign_summary,
        "merge_records_raw": capturer.merge_records,
    }
    with open(out_dir / "attn_propagation.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\njson dumped to {out_dir / 'attn_propagation.json'}")

    for h in q_handles + pre_handles:
        h.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
