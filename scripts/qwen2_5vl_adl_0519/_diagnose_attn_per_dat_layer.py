#!/usr/bin/env python
"""Per-DAT-layer HD cross-attention score diagnostic.

For each of the 6 DAT layers, capture pass-2 cross-attention scores
softmax(Q[ans] · K_HD^T / sqrt(d)) and report distribution quality:

  - hd_max_mean       : average of max attn weight per (head, ans-token)
  - hd_top10_mass     : average mass in top-10 HD tokens per row
  - hd_entropy_mean   : average row entropy (low = peaked, high = diffuse)
  - hd_uniform_entropy: log(Ns) baseline (fully uniform)
  - hd_entropy_norm   : entropy_mean / uniform_entropy (∈[0,1]; 1 = noise)
  - hd_ns             : number of HD-sampled tokens (sanity)
  - w_HD_mean         : LSE merge weight (existing metric, for context)
  - hd_gate_sigmoid   : σ(hd_gate) per layer (existing)

How: monkey-patch
  (a) each DAT module's `_generate_offsets_and_sample`  → set cur_dat_layer
                                                        + capture image_range_list
  (b) module-level `_dat_cross_attn_varlen`              → intercept q/k lists,
                                                          recompute attn scores
  (c) each DAT module's `_merge_two_pass_lse`            → w_HD + hd_gate
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import llava.model.language_model.modeling_qwen2_5vl_dat as _dat_mod
from llava.model.language_model.modeling_qwen2_5vl_dat import (
    Qwen2_5_VLAttentionDAT,
)


# ============================================================================
# Capture
# ============================================================================
class HDAttnCapture:
    def __init__(self):
        self.image_range_list: list | None = None
        self.cur_dat_layer: int | None = None
        self.current_sample_idx: int = 0
        # HD pass-2 records (sample × layer)
        self.records: list[dict] = []
        # LR pass-1 records (sample × layer): bucket masses + lr-block peakedness
        self.lr_records: list[dict] = []
        # for w_HD / hd_gate context
        self.merge_records: list[dict] = []
        self.n_heads = 0
        self.head_dim = 0

    def reset_sample(self):
        self.image_range_list = None
        self.cur_dat_layer = None

    # ----- install ------------------------------------------------------
    def install(self, model):
        dat_modules: list[tuple[int, Qwen2_5_VLAttentionDAT]] = []
        for name, mod in model.named_modules():
            if isinstance(mod, Qwen2_5_VLAttentionDAT):
                try:
                    L = int(name.split("layers.")[1].split(".")[0])
                except Exception:
                    continue
                dat_modules.append((L, mod))
                if self.n_heads == 0:
                    self.n_heads = int(mod.num_heads)
                    self.head_dim = int(mod.head_dim)
        dat_modules.sort(key=lambda x: x[0])

        for L, mod in dat_modules:
            self._patch_forward(mod, L)   # MUST run first: sets cur_dat_layer
            self._patch_meta(mod, L)
            self._patch_merge(mod, L)

        self._patch_cross_attn()
        self._patch_lr_attn()

        return [L for L, _ in dat_modules]

    # ----- patches ------------------------------------------------------
    def _patch_forward(self, attn_mod, layer_idx):
        """Set cur_dat_layer + capture image_range_list at the very start of
        each DAT module's forward, so both pass-1 (LR) and pass-2 (HD) are
        attributed to the right layer AND pass-1 (which runs BEFORE
        `_generate_offsets_and_sample`) has access to image_range_list.

        Signature of Qwen2_5_VLAttentionDAT.forward (after `self`):
            (hidden_states, attention_mask, position_ids, past_key_values,
             output_attentions, use_cache, cache_position, position_embeddings,
             image_hd_features, image_range_list, mrope_position_ids, **kwargs)
        → image_range_list is positional index 9 or kwarg.
        """
        orig = attn_mod.forward
        cap = self

        def patched(*args, **kwargs):
            cap.cur_dat_layer = layer_idx
            irl = kwargs.get("image_range_list", None)
            if irl is None and len(args) >= 10:
                irl = args[9]
            if irl is not None and cap.image_range_list is None:
                cap.image_range_list = irl
            return orig(*args, **kwargs)

        attn_mod.forward = patched

    def _patch_meta(self, attn_mod, layer_idx):
        """Set cur_dat_layer + capture image_range_list."""
        orig = attn_mod._generate_offsets_and_sample
        cap = self

        def patched(query_states, image_hd_features, image_range_list,
                    b_idx, hd_feat_idx):
            cap.cur_dat_layer = layer_idx
            if cap.image_range_list is None and image_range_list is not None:
                cap.image_range_list = image_range_list
            return orig(query_states, image_hd_features, image_range_list,
                        b_idx, hd_feat_idx)

        attn_mod._generate_offsets_and_sample = patched

    def _patch_merge(self, attn_mod, layer_idx):
        """Capture σ(hd_gate) and w_HD per call."""
        orig = attn_mod._merge_two_pass_lse
        cap = self

        def patched(out1, lse1, out2, lse2, ans_start, ans_end):
            with torch.no_grad():
                lse1_ans = lse1[:, :, ans_start:ans_end].float()
                lse2_f = lse2.float()
                if attn_mod.hd_gate is not None:
                    gate_log = F.logsigmoid(attn_mod.hd_gate.detach()).float()
                    hd_gate_val = float(attn_mod.hd_gate.detach().sigmoid().item())
                else:
                    gate_log = torch.zeros((), device=lse2_f.device, dtype=lse2_f.dtype)
                    hd_gate_val = 1.0
                lse2_gated = lse2_f + gate_log
                lse_total = torch.logaddexp(lse1_ans, lse2_gated)
                w2 = (lse2_gated - lse_total).exp()
                w_HD_mean = float(w2.mean().item())

            cap.merge_records.append({
                "sample_idx": cap.current_sample_idx,
                "layer_idx": layer_idx,
                "hd_gate_sigmoid": hd_gate_val,
                "w_HD_mean": w_HD_mean,
            })
            return orig(out1, lse1, out2, lse2, ans_start, ans_end)

        attn_mod._merge_two_pass_lse = patched

    def _patch_lr_attn(self):
        """Intercept _dat_attn_with_lse (LR pass-1) to compute bucket masses
        + LR-image peakedness for answer rows."""
        orig = _dat_mod._dat_attn_with_lse
        cap = self

        def patched(q, k, v, causal=True):
            layer_idx = cap.cur_dat_layer
            if layer_idx is not None and cap.image_range_list is not None:
                with torch.no_grad():
                    irl_b0 = cap.image_range_list[0]
                    if len(irl_b0) >= 2:
                        lr_start, lr_end, _lh, _lw = irl_b0[0]
                        ans_s_lbl, ans_e_lbl, intention = irl_b0[1]
                        # q/k: [B, H, Nq, D] post-RoPE, post-repeat_kv
                        B, H, Nq, D = q.shape
                        if ans_e_lbl > 0:
                            qa_start, qa_end = ans_s_lbl, ans_e_lbl
                        else:
                            qa_start, qa_end = intention + 1, Nq
                        qa_end = min(qa_end, Nq)
                        if qa_start < qa_end and qa_start < Nq:
                            Nans = qa_end - qa_start
                            q_ans = q[0, :, qa_start:qa_end, :].float().cpu()  # [H, Nans, D]
                            k_all = k[0].float().cpu()                          # [H, Nq, D]
                            scores = torch.einsum(
                                "hnd,hsd->hns", q_ans, k_all
                            ) * (1.0 / math.sqrt(D))                            # [H, Nans, Nq]
                            # causal mask: ans token at qa_start+r → cols [0, qa_start+r]
                            col = torch.arange(Nq).view(1, 1, -1)
                            row_max = (qa_start + torch.arange(Nans)).view(1, -1, 1)
                            mask = (col <= row_max).expand(H, -1, -1)
                            scores = scores.masked_fill(~mask, float("-inf"))
                            attn = torch.softmax(scores, dim=-1)                # [H, Nans, Nq]

                            attn_avg = attn.mean(dim=0)                         # [Nans, Nq]
                            mass_sys = (float(attn_avg[:, :lr_start].sum(-1).mean())
                                        if lr_start > 0 else 0.0)
                            mass_lr  = float(attn_avg[:, lr_start:lr_end].sum(-1).mean())
                            mass_qa  = (float(attn_avg[:, lr_end:qa_start].sum(-1).mean())
                                        if qa_start > lr_end else 0.0)
                            mass_self = max(0.0, 1.0 - mass_sys - mass_lr - mass_qa)

                            # peakedness within lr_image bucket (renormalize over Nlr)
                            lr_block = attn[:, :, lr_start:lr_end]              # [H, Nans, Nlr]
                            Nlr = int(lr_block.shape[-1])
                            if Nlr > 0:
                                lr_sum = lr_block.sum(-1, keepdim=True).clamp_min(1e-12)
                                lr_norm = lr_block / lr_sum
                                lr_max = float(lr_norm.max(-1).values.mean().item())
                                ktop = min(10, Nlr)
                                lr_top10 = float(
                                    lr_norm.topk(ktop, -1).values.sum(-1).mean().item()
                                )
                                lr_ent = float(
                                    -(lr_norm * (lr_norm + 1e-12).log()).sum(-1).mean().item()
                                )
                                lr_unif = math.log(Nlr) if Nlr > 1 else 0.0
                                lr_ent_norm = lr_ent / lr_unif if lr_unif > 0 else 1.0
                            else:
                                lr_max = lr_top10 = lr_ent = lr_unif = 0.0
                                lr_ent_norm = 1.0

                            cap.lr_records.append({
                                "sample_idx": cap.current_sample_idx,
                                "layer_idx": layer_idx,
                                "n_ans": Nans,
                                "n_lr": Nlr,
                                "mass_system": mass_sys,
                                "mass_lr_image": mass_lr,
                                "mass_qa_prefix": mass_qa,
                                "mass_ans_self": mass_self,
                                "lr_max_in_image": lr_max,
                                "lr_top10_in_image": lr_top10,
                                "lr_entropy_in_image": lr_ent,
                                "lr_uniform_in_image": lr_unif,
                                "lr_entropy_norm_in_image": lr_ent_norm,
                            })
            return orig(q, k, v, causal=causal)

        _dat_mod._dat_attn_with_lse = patched

    def _patch_cross_attn(self):
        """Intercept _dat_cross_attn_varlen to recompute per-segment attn."""
        orig = _dat_mod._dat_cross_attn_varlen
        cap = self

        def patched(q_list, k_list, v_list):
            layer_idx = cap.cur_dat_layer
            if layer_idx is not None:
                with torch.no_grad():
                    for seg_i, (q_seg, k_seg) in enumerate(zip(q_list, k_list)):
                        # q_seg: [Nans, H, D];  k_seg: [Ns, H, D]
                        H = int(q_seg.shape[1])
                        D = int(q_seg.shape[2])
                        Nans = int(q_seg.shape[0])
                        Ns = int(k_seg.shape[0])
                        if Nans == 0 or Ns == 0:
                            continue
                        # Use fp32 on CPU to keep GPU light. The compute is small.
                        q = q_seg.detach().float().to("cpu")
                        k = k_seg.detach().float().to("cpu")
                        # scores: [H, Nans, Ns]
                        scores = torch.einsum(
                            "nhd,shd->hns", q, k
                        ) * (1.0 / math.sqrt(D))
                        attn = torch.softmax(scores, dim=-1)
                        # metrics
                        max_per = attn.max(dim=-1).values            # [H, Nans]
                        eps = 1e-12
                        ent_per = -(attn * (attn + eps).log()).sum(dim=-1)
                        topk = min(10, Ns)
                        top10_mass = attn.topk(topk, dim=-1).values.sum(dim=-1)
                        uniform_ent = math.log(Ns) if Ns > 1 else 0.0

                        cap.records.append({
                            "sample_idx": cap.current_sample_idx,
                            "layer_idx": layer_idx,
                            "seg_idx": seg_i,
                            "n_ans": Nans,
                            "n_hd": Ns,
                            "hd_max_mean": float(max_per.mean().item()),
                            "hd_max_p95": float(max_per.quantile(0.95).item()),
                            "hd_entropy_mean": float(ent_per.mean().item()),
                            "hd_uniform_entropy": uniform_ent,
                            "hd_entropy_norm": (
                                float(ent_per.mean().item()) / uniform_ent
                                if uniform_ent > 0 else 1.0
                            ),
                            "hd_top10_mass_mean": float(top10_mass.mean().item()),
                        })
            return orig(q_list, k_list, v_list)

        _dat_mod._dat_cross_attn_varlen = patched


# ============================================================================
# Aggregate + print
# ============================================================================
def aggregate(cap: HDAttnCapture, dat_layers: list[int]) -> dict[int, dict]:
    by_layer = {L: [] for L in dat_layers}
    for r in cap.records:
        by_layer.setdefault(r["layer_idx"], []).append(r)
    merge_by_layer = {L: [] for L in dat_layers}
    for r in cap.merge_records:
        merge_by_layer.setdefault(r["layer_idx"], []).append(r)
    lr_by_layer = {L: [] for L in dat_layers}
    for r in cap.lr_records:
        lr_by_layer.setdefault(r["layer_idx"], []).append(r)

    out = {}
    for L in dat_layers:
        recs = by_layer.get(L, [])
        mrecs = merge_by_layer.get(L, [])
        lrrecs = lr_by_layer.get(L, [])
        if not recs:
            continue
        agg = {
            "n_samples": len(recs),
            "n_hd": int(np.mean([r["n_hd"] for r in recs])),
            "hd_max_mean":      float(np.mean([r["hd_max_mean"]       for r in recs])),
            "hd_max_p95":       float(np.mean([r["hd_max_p95"]        for r in recs])),
            "hd_entropy_mean":  float(np.mean([r["hd_entropy_mean"]   for r in recs])),
            "hd_uniform_ent":   float(np.mean([r["hd_uniform_entropy"] for r in recs])),
            "hd_entropy_norm":  float(np.mean([r["hd_entropy_norm"]   for r in recs])),
            "hd_top10_mass":    float(np.mean([r["hd_top10_mass_mean"] for r in recs])),
        }
        if mrecs:
            agg["hd_gate_sigmoid"] = float(np.mean([r["hd_gate_sigmoid"] for r in mrecs]))
            agg["w_HD_mean"]       = float(np.mean([r["w_HD_mean"]       for r in mrecs]))
        if lrrecs:
            agg["mass_system"]    = float(np.mean([r["mass_system"]    for r in lrrecs]))
            agg["mass_lr_image"]  = float(np.mean([r["mass_lr_image"]  for r in lrrecs]))
            agg["mass_qa_prefix"] = float(np.mean([r["mass_qa_prefix"] for r in lrrecs]))
            agg["mass_ans_self"]  = float(np.mean([r["mass_ans_self"]  for r in lrrecs]))
            agg["lr_max"]         = float(np.mean([r["lr_max_in_image"]         for r in lrrecs]))
            agg["lr_top10"]       = float(np.mean([r["lr_top10_in_image"]       for r in lrrecs]))
            agg["lr_ent_norm"]    = float(np.mean([r["lr_entropy_norm_in_image"] for r in lrrecs]))
            agg["n_lr"]           = int(np.mean([r["n_lr"] for r in lrrecs]))
        out[L] = agg
    return out


def print_table(summary: dict[int, dict], dat_layers: list[int], title: str):
    # ----- Table 1: where answer Q attends in LR pass-1 -------------------
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print("\n[A] LR pass-1 bucket masses on answer-token rows (sum=1 per row)")
    print(" layer | σ(gate) | w_HD   | sys    | lr_img | qa_pfx | ans_self")
    print("-" * 100)
    for L in sorted(dat_layers):
        s = summary.get(L)
        if s is None:
            continue
        print(
            f"  L{L:2d}  | {s.get('hd_gate_sigmoid', float('nan')):.4f}  |"
            f" {s.get('w_HD_mean', float('nan')):.4f} |"
            f" {s.get('mass_system', float('nan')):.4f} |"
            f" {s.get('mass_lr_image', float('nan')):.4f} |"
            f" {s.get('mass_qa_prefix', float('nan')):.4f} |"
            f" {s.get('mass_ans_self', float('nan')):.4f}"
        )

    # ----- Table 2: LR vs HD attention peakedness --------------------------
    print("\n[B] LR-image bucket vs HD attn peakedness (renormalized within bucket)")
    print(" layer |  LR_max | LR_top10 | LR_entN | N_lr || HD_max | HD_top10 | HD_entN | N_hd")
    print("-" * 100)
    for L in sorted(dat_layers):
        s = summary.get(L)
        if s is None:
            continue
        print(
            f"  L{L:2d}  | {s.get('lr_max', float('nan')):.4f}  |"
            f"  {s.get('lr_top10', float('nan')):.4f}  |"
            f"  {s.get('lr_ent_norm', float('nan')):.4f} |  {s.get('n_lr', 0):3d}"
            f" || {s['hd_max_mean']:.4f} |  {s['hd_top10_mass']:.4f}  |"
            f"  {s['hd_entropy_norm']:.4f} |  {s['n_hd']:3d}"
        )
    print("\nInterpretation:")
    print("  [A] mass_lr_image high → answer Q looks at LR image positions")
    print("      mass_qa_prefix high → answer Q looks at question text")
    print("  [B] max/top10 large + entN small → attention peaked on few positions")
    print("      LR vs HD comparison is apples-to-apples: both renormalized within bucket")


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device_map", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] instantiating Qwen2_5_DATVL adapter from {args.ckpt}")
    from lmms_eval.models.simple.qwen2_5_dat_vl import Qwen2_5_DATVL
    adapter = Qwen2_5_DATVL(
        pretrained=args.ckpt,
        device=args.device,
        device_map=args.device_map or args.device,
        batch_size=1,
        use_cache=True,
    )
    model = adapter._model

    print("[2/5] installing patches (meta + merge + cross_attn)")
    cap = HDAttnCapture()
    dat_layers = cap.install(model)
    print(f"  DAT layers: {dat_layers}")
    print(f"  H={cap.n_heads}, D={cap.head_dim}")

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

    print("[4/5] running generation + capturing HD attn per DAT layer")
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

    for i, idx in enumerate(chosen):
        doc = ds[int(idx)]
        try:
            _ = decode_base64_to_image(doc["image"])
        except Exception as e:
            print(f"  [WARN] sample {idx} image decode failed: {e}")
            continue
        prompt_text = hrbench_doc_to_text(doc)
        inst = Instance(
            request_type="generate_until",
            arguments=(prompt_text, {"max_new_tokens": 1, "temperature": 0.0},
                       hrbench_doc_to_visual, int(idx), TASK, SPLIT),
            idx=int(idx),
            metadata={"task": TASK, "doc_id": int(idx), "repeats": 0},
        )
        inst.doc = doc
        cap.current_sample_idx = i
        cap.reset_sample()
        try:
            with torch.no_grad():
                _ = adapter.generate_until([inst])
        except Exception as e:
            print(f"  [WARN] sample {idx} forward failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue
        if (i + 1) % 4 == 0:
            print(f"  processed {i + 1}/{len(chosen)}")

    print("[5/5] aggregating + writing outputs")
    summary = aggregate(cap, dat_layers)
    print_table(summary, dat_layers,
                f"Per-DAT-layer HD cross-attention quality  ({args.ckpt})")

    json_out = {
        "ckpt": args.ckpt,
        "dat_layers": dat_layers,
        "summary": {f"L{L}": v for L, v in summary.items()},
        "raw_records": cap.records,
    }
    with open(out_dir / "attn_per_dat_layer.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\njson dumped to {out_dir / 'attn_per_dat_layer.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
