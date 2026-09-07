#!/usr/bin/env python3
"""Diagnose and plot the sampling data dumped by vis_question_sampling.py.

The point of this pass is to answer one question before any paper figure gets
drawn: do the sampling locations actually move when the question changes, by an
amount that is large compared to the natural spread across DAT layers?

Per image it writes
  overview.png   one row per question: sampling points over the image, plus the
                 intention->LR attention guide that drives them
  stats.json     centroid shifts between questions, layer-wise spread as the
                 null baseline, and offset magnitude vs the reference grid pitch
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None


def to_pixels(locs, w, h):
    """[..., 2] normalized (y, x) in [-1,1] -> (x_px, y_px) on a w*h image."""
    y = (locs[..., 0] + 1.0) * 0.5 * h
    x = (locs[..., 1] + 1.0) * 0.5 * w
    return x, y


def load_image_dir(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    image = Image.open(os.path.join(d, "image.png"))
    qs = []
    for q in meta["questions"]:
        z = np.load(os.path.join(d, f"q{q['idx']}.npz"))
        qs.append({**q, "locs": z["locs"], "layers": z["layers"],
                   "guide": z["guide"] if "guide" in z else None})
    return meta, image, qs


def compute_stats(qs):
    """Question-driven movement vs the layer-to-layer null."""
    # centroid per (question, layer) in normalized coords
    cent = np.stack([q["locs"].mean(axis=1) for q in qs])       # [Q, L, 2]
    q_cent = cent.mean(axis=1)                                   # [Q, 2]

    # signal: how far apart are different questions
    pair = []
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            pair.append({
                "q": [i, j],
                "centroid_dist": float(np.linalg.norm(q_cent[i] - q_cent[j])),
                # per-layer, so a question shift is not washed out by averaging
                "per_layer_dist": float(np.linalg.norm(cent[i] - cent[j], axis=-1).mean()),
            })

    # null: spread of a single question's centroid across layers
    layer_spread = float(np.linalg.norm(cent - cent.mean(axis=1, keepdims=True),
                                        axis=-1).mean())

    # how far points travel from the uniform reference grid
    g = int(round(np.sqrt(qs[0]["locs"].shape[1])))
    pitch = 2.0 / max(g - 1, 1)
    spread = float(np.mean([q["locs"].reshape(-1, 2).std(axis=0).mean() for q in qs]))

    sig = float(np.mean([p["per_layer_dist"] for p in pair])) if pair else 0.0
    return {
        "n_question": len(qs), "grid": g, "ref_pitch": pitch,
        "point_spread_norm": spread,
        "pairwise": pair,
        "mean_question_shift": sig,
        "layer_null_spread": layer_spread,
        "signal_to_null": float(sig / layer_spread) if layer_spread > 0 else None,
    }


def plot_image(meta, image, qs, out_png, max_side=1400):
    scale = min(1.0, max_side / max(image.size))
    thumb = image.resize((max(1, int(image.width * scale)),
                          max(1, int(image.height * scale))), Image.LANCZOS)
    W, H = thumb.size

    n = len(qs)
    fig, axes = plt.subplots(n, 2, figsize=(15, 4.2 * n),
                             gridspec_kw={"width_ratios": [2.4, 1]})
    axes = np.atleast_2d(axes)

    for r, q in enumerate(qs):
        ax = axes[r, 0]
        ax.imshow(thumb)
        L = q["locs"].shape[0]
        cmap = plt.get_cmap("turbo")
        for li in range(L):
            x, y = to_pixels(q["locs"][li].reshape(-1, 2), W, H)
            ax.scatter(x, y, s=7, alpha=0.45, linewidths=0,
                       color=cmap(li / max(L - 1, 1)),
                       label=f"layer {q['layers'][li]}" if r == 0 else None)
        ok = "OK" if q["correct"] else "WRONG"
        ax.set_title(f"[{q['category']}] {q['question'][:78]}\n"
                     f"pred={q['pred'][:24]}  gt={q['gt']}  ({ok})", fontsize=9)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

        ax2 = axes[r, 1]
        if q["guide"] is not None:
            gm = q["guide"].mean(axis=0)              # mean over layers
            gm = gm.reshape(-1, *gm.shape[-2:])[0]    # first (only) answer slot
            ax2.imshow(gm, cmap="inferno")
            ax2.set_title("intention->LR attention (layer mean)", fontsize=9)
        ax2.axis("off")

    if n and axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(fontsize=6, markerscale=1.6, loc="lower right", ncol=2)
    fig.suptitle(f"{meta['hash']}  {meta['size'][0]}x{meta['size'][1]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=115, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="output dir of vis_question_sampling.py")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.run, "*"))
                  if os.path.exists(os.path.join(d, "meta.json")))
    summary = []
    for d in dirs:
        meta, image, qs = load_image_dir(d)
        if len(qs) < 2:
            continue
        st = compute_stats(qs)
        json.dump(st, open(os.path.join(d, "stats.json"), "w"), indent=1)
        plot_image(meta, image, qs, os.path.join(d, "overview.png"))
        summary.append({"hash": meta["hash"], **{k: st[k] for k in
                        ("n_question", "mean_question_shift", "layer_null_spread",
                         "signal_to_null", "point_spread_norm", "ref_pitch")}})
        print(f"{meta['hash']}: Q={st['n_question']} "
              f"question_shift={st['mean_question_shift']:.4f} "
              f"layer_null={st['layer_null_spread']:.4f} "
              f"ratio={st['signal_to_null']:.2f} "
              f"spread={st['point_spread_norm']:.4f} (pitch={st['ref_pitch']:.4f})",
              flush=True)

    json.dump(summary, open(os.path.join(args.run, "summary.json"), "w"), indent=1)
    print(f"-> {args.run}/summary.json", flush=True)


if __name__ == "__main__":
    main()
