#!/usr/bin/env python
"""Per-layer attention diagnostic for baseline Qwen2.5-VL (no DAT).

For each of the 36 LLM decoder layers, capture the FULL [B,H,Nq,Nq]
attention probabilities via attn_implementation="eager" + force
output_attentions=True per layer. Then bucket the answer-token rows into

  (system / lr_image / qa_prefix / ans_self)

so we can compare with the LR-pass-1 bucket masses from the DAT models.

Locating lr_image:
  scan input_ids for vision_start (151652) and vision_end (151653); lr_image
  = (vision_start_pos + 1, vision_end_pos).

Answer span:
  qa_start = Nq - 2, qa_end = Nq   (matches DAT's intention+1:Nq for
  generation mode with max_new_tokens=1, which is the same prompt structure).
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ============================================================================
# Capture
# ============================================================================
class BaselineAttnCapture:
    def __init__(self):
        self.attn_weights: dict[int, torch.Tensor] = {}  # layer_idx -> [B,H,Nq,Nq]
        self.input_ids: torch.Tensor | None = None
        self.current_sample_idx: int = 0
        self.records: list[dict] = []
        self.n_heads = 0
        self.head_dim = 0
        self.vision_start_id: int | None = None
        self.vision_end_id: int | None = None

    def reset_sample(self):
        self.attn_weights.clear()
        self.input_ids = None

    def install(self, model, tokenizer):
        # token IDs for image boundary
        try:
            self.vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            self.vision_end_id   = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        except Exception:
            pass

        # hook the LLM input embedding to capture input_ids
        embed = None
        for name, mod in model.named_modules():
            # Qwen2.5-VL text model: model.language_model.embed_tokens
            if name.endswith("language_model.embed_tokens") or name.endswith("model.embed_tokens"):
                embed = mod
                break
        if embed is not None:
            def _embed_pre(m, args, kwargs):
                ids = args[0] if args else kwargs.get("input")
                if ids is not None and ids.dtype == torch.long:
                    self.input_ids = ids.detach().cpu()
            embed.register_forward_pre_hook(_embed_pre, with_kwargs=True)
        else:
            print("[warn] could not locate embed_tokens; will fall back to input_ids=None")

        # patch every self_attn forward to force output_attentions=True and
        # capture attn_weights
        layer_indices = []
        for name, mod in model.named_modules():
            if not name.endswith(".self_attn"):
                continue
            if not hasattr(mod, "q_proj"):
                continue
            try:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
            except Exception:
                continue
            self._patch_attn(mod, layer_idx)
            layer_indices.append(layer_idx)
            if self.n_heads == 0 and hasattr(mod, "num_heads"):
                self.n_heads = int(mod.num_heads)
                self.head_dim = int(mod.head_dim)
        return sorted(set(layer_indices))

    def _patch_attn(self, attn_mod, layer_idx):
        orig = attn_mod.forward
        cap = self

        def patched(*args, **kwargs):
            kwargs["output_attentions"] = True
            out = orig(*args, **kwargs)
            # eager attention returns (attn_output, attn_weights[, past_key_value])
            attn_weights = None
            if isinstance(out, tuple) and len(out) >= 2:
                attn_weights = out[1]
            if attn_weights is not None:
                cap.attn_weights[layer_idx] = attn_weights.detach().float().cpu()
            return out

        attn_mod.forward = patched


# ============================================================================
# Bucketing
# ============================================================================
def find_image_range(input_ids: torch.Tensor, vs_id: int, ve_id: int) -> tuple[int, int] | None:
    """Find [lr_start, lr_end) in input_ids (shape [B, Nq], we use B=0)."""
    if input_ids is None or vs_id is None or ve_id is None:
        return None
    ids = input_ids[0].tolist()
    try:
        vs_pos = ids.index(vs_id)
        ve_pos = ids.index(ve_id)
    except ValueError:
        return None
    return (vs_pos + 1, ve_pos)


def compute_bucket_metrics(attn: torch.Tensor, lr_start: int, lr_end: int,
                            qa_start: int, qa_end: int) -> dict | None:
    """attn: [B, H, Nq, Nq]; we use B=0."""
    a = attn[0]                          # [H, Nq, Nq]
    H, Nq, _ = a.shape
    qa_end = min(qa_end, Nq)
    if qa_end <= qa_start or qa_start >= Nq:
        return None

    rows = a[:, qa_start:qa_end, :]      # [H, Nans, Nq]
    rows_avg = rows.mean(dim=0)          # [Nans, Nq]

    mass_sys = float(rows_avg[:, :lr_start].sum(-1).mean()) if lr_start > 0 else 0.0
    mass_lr  = float(rows_avg[:, lr_start:lr_end].sum(-1).mean())
    mass_qa  = (float(rows_avg[:, lr_end:qa_start].sum(-1).mean())
                if qa_start > lr_end else 0.0)
    mass_self = max(0.0, 1.0 - mass_sys - mass_lr - mass_qa)

    # peakedness within lr_image bucket
    lr_block = rows[:, :, lr_start:lr_end]      # [H, Nans, Nlr]
    Nlr = int(lr_block.shape[-1])
    if Nlr > 0:
        lr_sum = lr_block.sum(-1, keepdim=True).clamp_min(1e-12)
        lr_norm = lr_block / lr_sum
        lr_max = float(lr_norm.max(-1).values.mean().item())
        ktop = min(10, Nlr)
        lr_top10 = float(lr_norm.topk(ktop, -1).values.sum(-1).mean().item())
        lr_ent = float((-(lr_norm * (lr_norm + 1e-12).log()).sum(-1)).mean().item())
        lr_unif = math.log(Nlr) if Nlr > 1 else 0.0
        lr_ent_norm = lr_ent / lr_unif if lr_unif > 0 else 1.0
    else:
        lr_max = lr_top10 = lr_ent_norm = 0.0

    Nans = qa_end - qa_start
    return {
        "n_ans": Nans,
        "n_lr": Nlr,
        "mass_system": mass_sys,
        "mass_lr_image": mass_lr,
        "mass_qa_prefix": mass_qa,
        "mass_ans_self": mass_self,
        "lr_max": lr_max,
        "lr_top10": lr_top10,
        "lr_ent_norm": lr_ent_norm,
    }


# ============================================================================
# Aggregate + print
# ============================================================================
def aggregate(records, all_layers):
    by_layer = {L: [] for L in all_layers}
    for r in records:
        by_layer.setdefault(r["layer_idx"], []).append(r)
    out = {}
    for L in all_layers:
        recs = by_layer.get(L, [])
        if not recs:
            continue
        out[L] = {
            "n_samples": len(recs),
            "mass_system":    float(np.mean([r["mass_system"]    for r in recs])),
            "mass_lr_image":  float(np.mean([r["mass_lr_image"]  for r in recs])),
            "mass_qa_prefix": float(np.mean([r["mass_qa_prefix"] for r in recs])),
            "mass_ans_self":  float(np.mean([r["mass_ans_self"]  for r in recs])),
            "lr_max":         float(np.mean([r["lr_max"]         for r in recs])),
            "lr_top10":       float(np.mean([r["lr_top10"]       for r in recs])),
            "lr_ent_norm":    float(np.mean([r["lr_ent_norm"]    for r in recs])),
            "n_lr":           int(np.mean([r["n_lr"] for r in recs])),
            "n_ans":          int(np.mean([r["n_ans"] for r in recs])),
        }
    return out


def print_table(summary, all_layers, focus_layers, title):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print("\n[A] LR attention bucket masses on answer-token rows (sum=1 per row)")
    print(" layer | sys    | lr_img | qa_pfx | ans_self | (focus=DAT positions in DAT model)")
    print("-" * 110)
    for L in sorted(all_layers):
        s = summary.get(L)
        if s is None:
            continue
        mark = " *" if L in focus_layers else "  "
        print(
            f"  L{L:2d}{mark}| {s['mass_system']:.4f} |"
            f" {s['mass_lr_image']:.4f} | {s['mass_qa_prefix']:.4f} |"
            f" {s['mass_ans_self']:.4f}"
        )

    print("\n[B] LR-image peakedness on answer rows (renormalized within bucket)")
    print(" layer | LR_max | LR_top10 | LR_entN | N_lr")
    print("-" * 110)
    for L in sorted(all_layers):
        s = summary.get(L)
        if s is None:
            continue
        mark = " *" if L in focus_layers else "  "
        print(
            f"  L{L:2d}{mark}| {s['lr_max']:.4f} |"
            f"  {s['lr_top10']:.4f}  |"
            f"  {s['lr_ent_norm']:.4f} | {s['n_lr']:4d}"
        )


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device_map", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] instantiating Qwen2_5_VL (baseline) from {args.ckpt}")
    from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL
    adapter = Qwen2_5_VL(
        pretrained=args.ckpt,
        device=args.device,
        device_map=args.device_map or args.device,
        batch_size=1,
        attn_implementation="eager",   # MUST be eager for output_attentions
        use_cache=True,
    )
    model = adapter._model
    tokenizer = adapter._tokenizer

    print("[2/5] installing patches (force output_attentions, capture attn_weights)")
    cap = BaselineAttnCapture()
    all_layers = cap.install(model, tokenizer)
    print(f"  patched {len(all_layers)} self_attn modules")
    print(f"  H={cap.n_heads}, D={cap.head_dim}, "
          f"vs_id={cap.vision_start_id}, ve_id={cap.vision_end_id}")

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

    print("[4/5] running generation + capturing per-layer attn_weights")
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

        # Resolve image range for THIS sample
        img_range = find_image_range(cap.input_ids, cap.vision_start_id, cap.vision_end_id)
        if img_range is None:
            print(f"  [WARN] sample {idx} could not locate vision_start/end tokens")
            continue
        lr_start, lr_end = img_range
        # answer span: last 2 tokens (matches DAT diagnostic convention)
        # Use Nq from any captured attn (all layers have same Nq)
        if not cap.attn_weights:
            print(f"  [WARN] sample {idx} captured no attn_weights")
            continue
        any_attn = next(iter(cap.attn_weights.values()))
        Nq = int(any_attn.shape[-1])
        qa_start, qa_end = max(0, Nq - 2), Nq

        for L, attn in cap.attn_weights.items():
            metrics = compute_bucket_metrics(attn, lr_start, lr_end, qa_start, qa_end)
            if metrics is None:
                continue
            metrics["sample_idx"] = i
            metrics["layer_idx"] = L
            cap.records.append(metrics)

        if (i + 1) % 4 == 0:
            print(f"  processed {i + 1}/{len(chosen)}; "
                  f"Nq={Nq} lr=[{lr_start},{lr_end})")

    print("[5/5] aggregating + writing outputs")
    summary = aggregate(cap.records, all_layers)
    focus_layers = [0, 6, 12, 18, 24, 30]  # DAT layer positions
    print_table(summary, all_layers, focus_layers,
                f"Baseline Qwen2.5-VL per-layer LR attention  ({args.ckpt})")

    json_out = {
        "ckpt": args.ckpt,
        "all_layers": all_layers,
        "summary": {f"L{L}": v for L, v in summary.items()},
        "raw_records": cap.records,
    }
    with open(out_dir / "attn_per_layer_baseline.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\njson dumped to {out_dir / 'attn_per_layer_baseline.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
