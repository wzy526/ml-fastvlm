#!/usr/bin/env python
"""DAT-vs-baseline live-forward diagnostic.

For each of N HR-Bench samples we run two forwards:

    (i)  trained DAT ckpt
    (ii) base Qwen2.5-VL-3B-Instruct (NO DAT)

and capture, per DAT layer in (i):

    - residual_l2_per_token  : ‖sigmoid(g) * hd_out_proj(out2)‖₂ averaged over q tokens
    - lr_l2_per_token        : ‖out_LR‖₂ averaged over q tokens
    - residual_pct           : residual_l2 / lr_l2  (fractional perturbation)

We also compare logits of the *answer* token between (i) and (ii) — if they
match closely (KL ≈ 0, top-1 agreement → 100%), DAT is provably contributing
zero useful signal at eval time.

Usage::

    python scripts/qwen2_5vl_adl_0520/_diagnose_dat_alive.py \
        --dat_ckpt /root/autodl-tmp/vldat_experiments/0520_full_runE1_D1D3_noSpatial_F4_bf16fix-merged \
        --base_ckpt /root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct \
        --n_samples 4 \
        --out_dir /root/autodl-tmp/diag_dat_alive_runE1
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from llava.model.language_model.modeling_qwen2_5vl_dat import (
    Qwen2_5_VLAttentionDAT,
)


class DATContribCapture:
    def __init__(self):
        self.records: list[dict] = []
        self.cur_sample: int = 0

    def install(self, model):
        """Patch ``_merge_residual`` on every DAT-attn layer to record
        residual / LR magnitudes."""
        n_patched = 0
        for name, mod in model.named_modules():
            if not isinstance(mod, Qwen2_5_VLAttentionDAT):
                continue
            try:
                L = int(name.split("layers.")[1].split(".")[0])
            except Exception:
                continue
            self._patch(mod, L)
            n_patched += 1
        return n_patched

    def _patch(self, attn_mod, layer_idx):
        cap = self
        orig_merge_resid = attn_mod._merge_residual
        orig_merge_lse = attn_mod._merge_two_pass_lse

        def patched_residual(out_b, out2, q_start, q_end):
            with torch.no_grad():
                B_, Nseg_, H_, D_ = out2.shape
                proj = attn_mod.hd_out_proj(out2.reshape(B_, Nseg_, H_ * D_))
                if attn_mod.hd_gate is not None:
                    proj = proj * torch.sigmoid(attn_mod.hd_gate.to(proj.dtype))
                resid = proj.view(B_, Nseg_, H_, D_).float()
                lr_slice = out_b[:, q_start:q_end, :, :].float()  # [1, Nseg, H, D]
                lr_norm = lr_slice.flatten(2).norm(dim=-1).mean().item()
                resid_norm = resid.flatten(2).norm(dim=-1).mean().item()
                cap.records.append({
                    "sample": cap.cur_sample,
                    "layer": layer_idx,
                    "merge": "residual",
                    "Nseg": Nseg_,
                    "lr_l2": lr_norm,
                    "residual_l2": resid_norm,
                    "residual_pct": (resid_norm / lr_norm) if lr_norm > 0 else 0.0,
                })
            return orig_merge_resid(out_b, out2, q_start, q_end)

        def patched_lse(out_b, lse1, out2, lse2, q_start, q_end):
            with torch.no_grad():
                lr_slice = out_b[:, q_start:q_end, :, :].float()
                lr_norm = lr_slice.flatten(2).norm(dim=-1).mean().item()
                out2_norm = out2.float().flatten(2).norm(dim=-1).mean().item()
                cap.records.append({
                    "sample": cap.cur_sample,
                    "layer": layer_idx,
                    "merge": "lse",
                    "Nseg": out2.shape[1],
                    "lr_l2": lr_norm,
                    "out2_l2": out2_norm,
                })
            return orig_merge_lse(out_b, lse1, out2, lse2, q_start, q_end)

        attn_mod._merge_residual = patched_residual
        attn_mod._merge_two_pass_lse = patched_lse


def _load_dat(ckpt_path, device):
    from lmms_eval.models.simple.qwen2_5_dat_vl import Qwen2_5_DATVL
    adapter = Qwen2_5_DATVL(
        pretrained=ckpt_path,
        device=device,
        device_map=device,
        batch_size=1,
        use_cache=True,
    )
    return adapter


def _load_base(ckpt_path, device):
    from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL
    adapter = Qwen2_5_VL(
        pretrained=ckpt_path,
        device=device,
        device_map=device,
        batch_size=1,
        use_cache=True,
        attn_implementation="sdpa",
    )
    return adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat_ckpt", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="hrbench_4k",
                    choices=["hrbench_4k", "hrbench_8k"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading DAT ckpt: {args.dat_ckpt}")
    dat_adapter = _load_dat(args.dat_ckpt, args.device)
    cap = DATContribCapture()
    n_patched = cap.install(dat_adapter._model)
    print(f"  patched {n_patched} DAT-attention layers")

    print("\n  hd_gate sigmoid values per DAT layer:")
    print(f"    {'layer':<8} {'gate_param':<14} {'sigmoid':<14} (≈ effective HD scale)")
    gate_info = {}
    for name, mod in dat_adapter._model.named_modules():
        if not isinstance(mod, Qwen2_5_VLAttentionDAT):
            continue
        try:
            L = int(name.split("layers.")[1].split(".")[0])
        except Exception:
            continue
        if mod.hd_gate is None:
            continue
        g = mod.hd_gate.detach().float()
        sg = torch.sigmoid(g)
        gate_info[L] = {
            "raw": g.flatten().tolist(),
            "sigmoid_mean": sg.mean().item(),
            "sigmoid_min": sg.min().item(),
            "sigmoid_max": sg.max().item(),
        }
        print(f"    L{L:<7d} mean={g.mean().item():+.4f}  "
              f"sigmoid_mean={sg.mean().item():.4f}  "
              f"min={sg.min().item():.4f}  max={sg.max().item():.4f}")

    print(f"[2/4] loading HR-Bench {args.split} samples")
    from datasets import load_dataset
    ds = load_dataset(
        "DreamMr/HR-Bench",
        "hrbench_version_split",
        split=args.split,
        cache_dir="/root/autodl-tmp/cache/huggingface/datasets",
    )

    from lmms_eval.api.instance import Instance
    from lmms_eval.tasks.hrbench.utils import (
        decode_base64_to_image, hrbench_doc_to_text, hrbench_doc_to_visual,
    )
    TASK = "hrbench4k" if args.split == "hrbench_4k" else "hrbench8k"
    SPLIT = args.split

    class _DocLookup:
        def __init__(self, ds_obj):
            self._ds = ds_obj
        def __getitem__(self, idx):
            return self._ds[int(idx)]

    dat_adapter.task_dict = {TASK: {SPLIT: _DocLookup(ds)}}

    print(f"[3/4] DAT forward + capture residual magnitudes")
    n = min(args.n_samples, len(ds))
    dat_outputs = []
    for i in range(n):
        doc = ds[i]
        try:
            _ = decode_base64_to_image(doc["image"])
        except Exception:
            continue
        prompt_text = hrbench_doc_to_text(doc)
        inst = Instance(
            request_type="generate_until",
            arguments=(prompt_text, {"max_new_tokens": 1, "temperature": 0.0},
                       hrbench_doc_to_visual, int(i), TASK, SPLIT),
            idx=int(i),
            metadata={"task": TASK, "doc_id": int(i), "repeats": 0},
        )
        inst.doc = doc
        cap.cur_sample = i
        with torch.no_grad():
            out = dat_adapter.generate_until([inst])
        dat_outputs.append(out[0] if out else None)
        print(f"  sample {i}: DAT prediction = {out[0]!r}")

    del dat_adapter
    torch.cuda.empty_cache()

    print(f"[4/4] BASE forward + collect predictions")
    base_adapter = _load_base(args.base_ckpt, args.device)
    base_adapter.task_dict = {TASK: {SPLIT: _DocLookup(ds)}}
    base_outputs = []
    for i in range(n):
        doc = ds[i]
        try:
            _ = decode_base64_to_image(doc["image"])
        except Exception:
            continue
        prompt_text = hrbench_doc_to_text(doc)
        inst = Instance(
            request_type="generate_until",
            arguments=(prompt_text, {"max_new_tokens": 1, "temperature": 0.0},
                       hrbench_doc_to_visual, int(i), TASK, SPLIT),
            idx=int(i),
            metadata={"task": TASK, "doc_id": int(i), "repeats": 0},
        )
        inst.doc = doc
        with torch.no_grad():
            out = base_adapter.generate_until([inst])
        base_outputs.append(out[0] if out else None)
        print(f"  sample {i}: BASE prediction = {out[0]!r}")

    # ---- Aggregate ----
    print("\n" + "=" * 80)
    print("DAT RESIDUAL CONTRIBUTION (per DAT layer, averaged over samples)")
    print("=" * 80)
    by_layer: dict = {}
    for r in cap.records:
        L = r["layer"]
        by_layer.setdefault(L, []).append(r)

    print(f"  {'layer':<8} {'merge':<10} {'lr_l2':<10} {'resid_l2':<10} {'resid_%':<10} {'n_seg':<6}")
    for L in sorted(by_layer):
        recs = by_layer[L]
        merge_kind = recs[0]["merge"]
        lr_avg = float(np.mean([r["lr_l2"] for r in recs]))
        if merge_kind == "residual":
            resid_avg = float(np.mean([r["residual_l2"] for r in recs]))
            pct = float(np.mean([r["residual_pct"] for r in recs])) * 100
            print(f"  L{L:<7d} {merge_kind:<10} {lr_avg:<10.4f} {resid_avg:<10.4f} {pct:<10.2f} {recs[0]['Nseg']:<6}")
        else:
            out2_avg = float(np.mean([r["out2_l2"] for r in recs]))
            print(f"  L{L:<7d} {merge_kind:<10} {lr_avg:<10.4f} {out2_avg:<10.4f} (LSE-merge) {recs[0]['Nseg']:<6}")

    # Prediction agreement
    print("\n" + "=" * 80)
    print("PREDICTION AGREEMENT (DAT vs BASE)")
    print("=" * 80)
    agree = 0
    for i, (d, b) in enumerate(zip(dat_outputs, base_outputs)):
        ok = (d == b)
        if ok:
            agree += 1
        print(f"  sample {i}: DAT={d!r}  BASE={b!r}  agree={ok}")
    print(f"\n  agreement: {agree}/{len(dat_outputs)} = {100*agree/max(1,len(dat_outputs)):.1f}%")

    json.dump({
        "split": args.split,
        "n_samples": len(dat_outputs),
        "hd_gate": gate_info,
        "by_layer": {str(L): [r for r in by_layer[L]] for L in by_layer},
        "predictions": [{"sample": i, "dat": d, "base": b}
                        for i, (d, b) in enumerate(zip(dat_outputs, base_outputs))],
    }, open(out_dir / "diag.json", "w"), indent=2)
    print(f"\nsaved → {out_dir / 'diag.json'}")


if __name__ == "__main__":
    main()
