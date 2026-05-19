"""Quantitative diagnostic of DAT offset behavior on HR-Bench 4K.

Strategy: instantiate the lmms-eval DAT adapter (Qwen2_5_DATVL) so we get
exactly the same LR-first resize geometry as eval; install forward hooks
on every Qwen2_5_VLAttentionDAT.conv_off_proj; run the adapter's
generate_until on a few HR-Bench docs; capture offsets during the
generation prefill pass.

What we measure (per D-layer, aggregated over N samples):
  1. magnitude_mean_avg ~ how far do offsets shift from canonical grid in [-1,1]
                          < 0.05 => essentially uniform grid (offsets dead)
                          0.1-0.5 => meaningful learned offsets
                          > 0.5  => aggressive cross-image jumps
  2. magnitude_mean_std ~ does *average* offset magnitude vary across samples?
                          ~0  => same shift regardless of input
  3. elemwise_std_xsamp ~ element-wise variance of raw offsets across samples
                          ~0 => offsets are nearly identical across inputs
                                (network dead, fully collapsed)
                          near magnitude_mean_avg => fully input-conditional

Output:
  - Console summary
  - JSON dump of per-sample / per-layer stats
  - 6 PNG plots: 2D scatter of sample_locs over [-1,1] for visual confirmation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/root/autodl-tmp/lmms-eval")

# Speed up offline by disabling network checks
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from llava.model.language_model.modeling_qwen2_5vl_dat import (  # noqa: E402
    Qwen2_5_VLAttentionDAT,
)


# ============================================================================
# Hook
# ============================================================================
class OffsetCapture:
    """Hook conv_off_proj.forward and grab its output (= offsets, fp32)."""

    def __init__(self):
        self.captures: dict[int, torch.Tensor] = {}

    def __call__(self, module, inputs, output):
        layer_idx = getattr(module, "_dat_diag_layer_idx", id(module))
        # output: [Lp*off_grps, 2, gh, gw]
        self.captures[layer_idx] = output.detach().float().cpu()


def install_hooks(model):
    cap = OffsetCapture()
    handles = []
    layer_indices = []
    for name, mod in model.named_modules():
        if isinstance(mod, Qwen2_5_VLAttentionDAT):
            try:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
            except Exception:
                layer_idx = id(mod)
            mod.conv_off_proj._dat_diag_layer_idx = layer_idx
            handles.append(mod.conv_off_proj.register_forward_hook(cap))
            layer_indices.append(layer_idx)
    return cap, handles, sorted(layer_indices)


# ============================================================================
# Per-sample / cross-sample analysis
# ============================================================================
def analyze_capture(captures: dict[int, torch.Tensor]) -> dict:
    out = {}
    for layer_idx, off in captures.items():
        # off: [Lp*g, 2, gh, gw]
        mag = off.abs().mean(dim=1).flatten()
        out[layer_idx] = {
            "n_offsets": int(mag.numel()),
            "magnitude_mean": float(mag.mean()),
            "magnitude_std": float(mag.std()),
            "abs_max": float(off.abs().max()),
            "raw_offsets": off.numpy(),
        }
    return out


def cross_sample_analysis(per_sample, layer_indices):
    summary = {}
    for layer_idx in layer_indices:
        mags = np.array([s[layer_idx]["magnitude_mean"] for s in per_sample])
        # element-wise variance: take first off_grps groups (1 group, 2, gh, gw)
        # so all samples are shape-compatible.
        raws = []
        for s in per_sample:
            r = s[layer_idx]["raw_offsets"]
            raws.append(r[:1])  # [1, 2, gh, gw]
        try:
            stacked = np.stack(raws, axis=0)
            elem_std = float(stacked.std(axis=0).mean())
        except Exception:
            elem_std = float("nan")
        summary[layer_idx] = {
            "samples_n": int(len(mags)),
            "magnitude_mean_avg": float(mags.mean()),
            "magnitude_mean_std": float(mags.std()),
            "elementwise_std_across_samples": elem_std,
        }
    return summary


def plot_sample_locs(per_sample, layer_indices, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable, skipping plots")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer_idx in layer_indices:
        fig, ax = plt.subplots(figsize=(6, 6))
        for s_idx, s in enumerate(per_sample):
            r = s[layer_idx]["raw_offsets"]
            gh, gw = r.shape[-2], r.shape[-1]
            ys = np.linspace(-1, 1, gh)
            xs = np.linspace(-1, 1, gw)
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            mean_off = r.mean(axis=0)  # [2, gh, gw]
            samp_y = (grid_y + mean_off[0]).clip(-1, 1).flatten()
            samp_x = (grid_x + mean_off[1]).clip(-1, 1).flatten()
            ax.scatter(samp_x, samp_y, alpha=0.25, s=8,
                       label=f"s{s_idx}" if s_idx < 6 else None)
        gh, gw = r.shape[-2], r.shape[-1]
        ys = np.linspace(-1, 1, gh)
        xs = np.linspace(-1, 1, gw)
        ax.scatter(*np.meshgrid(xs, ys, indexing="xy"),
                   c="red", marker="x", s=15, label="canonical")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_title(f"Layer {layer_idx} — sample_locs (N={len(per_sample)})")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        fig.savefig(out_dir / f"layer{layer_idx:02d}_sample_locs.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"saved scatter plots to {out_dir}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/root/autodl-tmp/vldat_experiments/0515_full_1d5l_sa1b_ivcap_runB_spatial_warmup_rmsnorm-merged",
    )
    parser.add_argument("--n_samples", type=int, default=12)
    parser.add_argument("--out_dir", default="/root/autodl-tmp/diag_offsets_runB")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] instantiating Qwen2_5_DATVL adapter from {args.ckpt}")
    # Build the adapter the same way lmms-eval would, but in single-process mode.
    from lmms_eval.models.simple.qwen2_5_dat_vl import Qwen2_5_DATVL
    adapter = Qwen2_5_DATVL(
        pretrained=args.ckpt,
        device=args.device,
        device_map=args.device,
        batch_size=1,
        attn_implementation="sdpa",
    )

    print("[2/5] installing offset capture hooks")
    capturer, handles, layer_indices = install_hooks(adapter._model)
    print(f"  hooks attached to layers: {layer_indices}")

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

    print("[4/5] running generation + capturing offsets")
    from lmms_eval.api.instance import Instance
    from lmms_eval.tasks.hrbench.utils import (
        decode_base64_to_image, hrbench_doc_to_text, hrbench_doc_to_visual,
    )

    # The adapter's generate_until calls
    #     visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) ...]
    # so we build a tiny lookup dict[idx] -> doc, mirrored as
    # task_dict[TASK][SPLIT][idx] = doc.
    TASK = "hrbench4k"
    SPLIT = "hrbench_4k"

    class _DocLookup:
        def __init__(self, ds_obj):
            self._ds = ds_obj
        def __getitem__(self, idx):
            return self._ds[int(idx)]

    adapter.task_dict = {TASK: {SPLIT: _DocLookup(ds)}}

    per_sample = []
    sample_meta = []
    for i, idx in enumerate(chosen):
        doc = ds[int(idx)]
        try:
            _ = decode_base64_to_image(doc["image"])  # sanity
        except Exception as e:
            print(f"  [WARN] sample {idx} image decode failed: {e}")
            continue

        # Build a generate_until-compatible Instance.
        # Args layout per lmms-eval source: (context_str, gen_kwargs, doc_to_visual_fn,
        #                                    doc_idx, task, split)
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

        capturer.captures.clear()
        try:
            with torch.no_grad():
                _ = adapter.generate_until([instance])
        except Exception as e:
            print(f"  [WARN] sample {idx} forward failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

        if not capturer.captures:
            print(f"  [WARN] sample {idx}: no offsets captured")
            continue
        stats = analyze_capture(capturer.captures)
        per_sample.append(stats)
        sample_meta.append({
            "idx": int(idx),
            "category": str(doc.get("category", "?")),
            "question": doc["question"][:80],
        })
        if (i + 1) % 4 == 0:
            print(f"  processed {i + 1}/{len(chosen)}")

    if not per_sample:
        print("[ERR] no samples produced offsets, aborting")
        sys.exit(1)

    print(f"[5/5] cross-sample analysis ({len(per_sample)} valid samples)")
    summary = cross_sample_analysis(per_sample, layer_indices)

    print("\n" + "=" * 80)
    print("DAT Offset Diagnostic Summary")
    print("=" * 80)
    print(f"Run B ckpt: {args.ckpt}")
    print(f"# valid samples: {len(per_sample)}")
    print()
    fmt = "{:>10s} | {:>14s} | {:>14s} | {:>20s}"
    print(fmt.format("layer", "magn_mean_avg", "magn_mean_std", "elemwise_std_xsamp"))
    print("-" * 80)
    for layer_idx in layer_indices:
        s = summary[layer_idx]
        print(fmt.format(
            f"L{layer_idx}",
            f"{s['magnitude_mean_avg']:.4f}",
            f"{s['magnitude_mean_std']:.4f}",
            f"{s['elementwise_std_across_samples']:.4f}",
        ))
    print()
    print("Interpretation:")
    print("  magn_mean_avg < 0.05  => essentially uniform grid (offsets dead)")
    print("  magn_mean_avg 0.1-0.5 => meaningful learned offsets")
    print("  magn_mean_std ~ 0     => same offset magnitude regardless of input")
    print("  elemwise_std_xsamp ~ 0 => offsets identical across inputs (collapsed)")

    # Persist
    json_out = {
        "ckpt": args.ckpt,
        "n_samples": len(per_sample),
        "summary": {f"L{k}": v for k, v in summary.items()},
        "samples": sample_meta,
        "per_sample_magnitudes": [
            {f"L{k}": stats[k]["magnitude_mean"] for k in layer_indices}
            for stats in per_sample
        ],
    }
    with open(out_dir / "offset_diag.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"json dumped to {out_dir / 'offset_diag.json'}")

    plot_sample_locs(per_sample, layer_indices, out_dir / "plots")
    for h in handles:
        h.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
