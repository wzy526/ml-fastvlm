#!/usr/bin/env python3
"""Figure 1(c): the coupled sampler's gradient has no target direction.

Left   histogram of cos(dL_LM/dp_j, p_j - p_j*) over every sampling point of
       every DAT layer of every V* sample (from the shard jsons written by
       collapse_probe.py --grad); the decoupled head's gradient sits at +1 by
       construction (softmax - t on cell logits pulls straight at the target).
Middle one sample: the learned grid of one DAT layer on the image, arrows =
       descent direction of the LM loss (-dL_LM/dp_j), unit length.
Right  same grid, arrows = direction to the oracle grid (p_j* - p_j).
       GT box in green, oracle window dashed.

  python scripts/collapse_fig.py --dir $E/loc/collapse3 --out fig_collapse2.pdf
  options: --sample 0123 (npz stem in <dir>/grad_dump), --layer 11, --every 2
"""
import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

COS_BINS = 40


def load_rows(d):
    rows = []
    for fn in sorted(glob.glob(os.path.join(d, "*.json"))):
        rows += json.load(open(fn))
    return [r for r in rows if r.get("grad") and r["grad"].get("cos_hist")]


def norm_to_px(p, w, h):
    """[-1, 1] image-fraction coords -> pixels of a w x h raster."""
    return (p[..., 0] + 1) / 2 * w, (p[..., 1] + 1) / 2 * h


def quiver_panel(ax, img, pts, vec, box, tgt, title, color, every):
    h, w = img.shape[:2]
    ax.imshow(img)
    sel = np.arange(pts.shape[0])
    if every > 1:                                    # thin a 20x20 grid to every k-th row/col
        g = int(round(np.sqrt(pts.shape[0])))
        rr, cc = np.divmod(sel, g)
        sel = sel[(rr % every == 0) & (cc % every == 0)]
    x, y = norm_to_px(pts[sel], w, h)
    v = vec[sel] / (np.linalg.norm(vec[sel], axis=-1, keepdims=True) + 1e-12)
    ax.scatter(x, y, s=3, c="white", linewidths=0, alpha=0.8)
    # angles="xy": arrows live in data coords; imshow's inverted y keeps image-down = +y
    ax.quiver(x, y, v[:, 0], v[:, 1], color=color, angles="xy", scale_units="width", scale=14,
              width=0.006, headwidth=4, headlength=5, alpha=0.95)
    bx0, by0 = norm_to_px(np.array([box[0], box[1]])[None], w, h)
    bx1, by1 = norm_to_px(np.array([box[2], box[3]])[None], w, h)
    ax.add_patch(Rectangle((bx0[0], by0[0]), bx1[0] - bx0[0], by1[0] - by0[0],
                           fill=False, ec="lime", lw=1.5))
    tx, ty = norm_to_px(tgt, w, h)
    ax.add_patch(Rectangle((tx.min(), ty.min()), tx.max() - tx.min(), ty.max() - ty.min(),
                           fill=False, ec="lime", lw=1.0, ls="--"))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with shard jsons (+ grad_dump/)")
    ap.add_argument("--dump_dir", default=None)
    ap.add_argument("--sample", default=None, help="npz stem, e.g. 0123; default: first one")
    ap.add_argument("--layer", type=int, default=None, help="DAT layer for the quiver; default: middle one")
    ap.add_argument("--every", type=int, default=2, help="draw every k-th grid row/col")
    ap.add_argument("--slot", type=int, default=-1, help="which (l g) slot of the grid tensor")
    ap.add_argument("--out", default="fig_collapse2.pdf")
    args = ap.parse_args()
    d = os.path.expanduser(args.dir)
    rows = load_rows(d)
    if not rows:
        raise SystemExit(f"no rows with grad/cos_hist under {d} (rerun collapse_probe with --grad)")
    hist = np.sum([r["grad"]["cos_hist"] for r in rows], axis=0).astype(float)
    edges = np.linspace(-1, 1, COS_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    n_pts = hist.sum()
    cos_mean = float(np.mean([r["grad"]["cos_mean"] for r in rows]))
    pos_frac = float(np.mean([r["grad"]["cos_pos_frac"] for r in rows]))
    coh = float(np.mean([r["grad"]["coh_mean"] for r in rows]))
    ns = rows[0]["grad"].get("ns", 400)

    fig = plt.figure(figsize=(9.6, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax.bar(centres, hist / n_pts / (edges[1] - edges[0]), width=edges[1] - edges[0],
           color="#c0392b", alpha=0.75, label=r"coupled sampler: $\partial L_{LM}/\partial p_j$")
    ax.axvline(1.0, color="#1f77b4", lw=2.5)
    ax.text(0.96, 0.5, "decoupled head\n(by construction)", ha="right", va="center",
            color="#1f77b4", fontsize=8, transform=ax.get_xaxis_transform())
    ax.axvline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"$\cos(\partial L_{LM}/\partial p_j,\ p_j - p_j^*)$", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_xlim(-1.05, 1.05)
    ax.tick_params(labelsize=8)
    ax.text(0.03, 0.95, f"{len(rows)} samples, {int(n_pts):,} points\nmean {cos_mean:+.3f}, "
            f"toward target {100 * pos_frac:.1f}%\ncoherence {coh:.2f} (noise {1 / np.sqrt(ns):.2f})",
            transform=ax.transAxes, fontsize=7.5, va="top")
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    ax.set_title("(c) gradient on the sampling points", fontsize=9, loc="left")

    dump_dir = os.path.expanduser(args.dump_dir) if args.dump_dir else os.path.join(d, "grad_dump")
    files = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    if args.sample:
        files = [f for f in files if os.path.basename(f).startswith(args.sample)]
    if not files:
        print(f"[fig] no npz under {dump_dir}; writing the histogram only")
    else:
        z = np.load(files[0])
        layers = [int(x) for x in z["layers"]]
        li = args.layer if args.layer is not None else layers[len(layers) // 2]
        pts = z[f"grid_{li}"][args.slot].astype(np.float32)        # [Ns, 2]
        glm = z[f"glm_{li}"][args.slot].astype(np.float32)
        tgt = z["target_grid"].astype(np.float32)
        box = z["target_box"].astype(np.float32)
        img = plt.imread(files[0][:-4] + ".jpg")
        stem = os.path.basename(files[0])[:-4]
        quiver_panel(fig.add_subplot(gs[1]), img, pts, -glm, box, tgt,
                     f"V* #{int(stem)}, layer {li}: $-\\partial L_{{LM}}/\\partial p_j$", "#c0392b", args.every)
        quiver_panel(fig.add_subplot(gs[2]), img, pts, tgt - pts, box, tgt,
                     r"same grid: $p_j^* - p_j$ (what the head gives)", "#1f77b4", args.every)
        c = np.sum(glm * (pts - tgt), -1) / (np.linalg.norm(glm, axis=-1) * np.linalg.norm(pts - tgt, axis=-1) + 1e-12)
        print(f"[fig] {stem} layer {li}: cos mean {c.mean():+.3f}, toward {100 * (c > 0).mean():.1f}%, "
              f"available layers {layers}, slots {z[f'grid_{li}'].shape[0]}")
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
