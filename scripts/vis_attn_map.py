#!/usr/bin/env python3
"""Render the question-conditioned read map -- the Fig-1 candidate.

What is plotted is NOT the sampling locations (those barely move with the
question) but where the answer token actually *reads*: the attention weight the
last answer query places on each sampled HD key, splatted onto that key's source
pixels.

Each of the Ns keys concatenates features from off_grps positions, so attention
mass a_s is deposited at all grps source locations of key s.

Layer/head selection matters and is made explicit:
  --layer  index into the DAT layers (0 = shallowest, which carries the largest
           HD share, w_hd ~= 0.35)
  --heads  'sensitive' picks the heads whose read moves most across questions,
           'all' averages every head (weaker, but assumption-free)
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
GRPS = 8


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    image = Image.open(os.path.join(d, "image.png"))
    out = []
    for q in meta["questions"]:
        z = np.load(os.path.join(d, "q%d.npz" % q["idx"]))
        if "attn" not in z.files:
            continue
        Ns = z["attn"].shape[-1]
        out.append({
            "q": q,
            "attn": z["attn"][:, :, -1, :],                                  # [L, H, Ns]
            "locs": z["locs"].reshape(z["locs"].shape[0], GRPS, Ns, 2),      # [L, G, Ns, 2]
        })
    return meta, image, out


def sensitive_heads(recs, layer, k=4):
    """heads whose read centroid moves most between questions"""
    cs = []
    for r in recs:
        a = r["attn"][layer] / r["attn"][layer].sum(-1, keepdims=True)       # [H, Ns]
        p = r["locs"][layer].mean(0)                                         # [Ns, 2]
        cs.append(a @ p)                                                     # [H, 2]
    sh = np.zeros(cs[0].shape[0])
    n = 0
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            sh += np.linalg.norm(cs[i] - cs[j], axis=-1)
            n += 1
    return np.argsort(sh / max(n, 1))[::-1][:k]


def read_map(rec, layer, heads, shape, bins):
    """splat attention onto source pixels -> smoothed 2D map"""
    H, W = shape
    a = rec["attn"][layer][heads].mean(0)                 # [Ns]
    a = a / a.sum()
    p = rec["locs"][layer]                                # [G, Ns, 2]  (y, x)
    y = ((p[..., 0] + 1) * 0.5 * (H - 1)).ravel()
    x = ((p[..., 1] + 1) * 0.5 * (W - 1)).ravel()
    w = np.tile(a, (p.shape[0],))                         # same mass at each group's source
    hist, _, _ = np.histogram2d(
        y, x, bins=bins, range=[[0, H - 1], [0, W - 1]], weights=w)
    # light gaussian blur via separable box passes (no scipy dependency)
    for _ in range(2):
        hist = (hist + np.roll(hist, 1, 0) + np.roll(hist, -1, 0)) / 3.0
        hist = (hist + np.roll(hist, 1, 1) + np.roll(hist, -1, 1)) / 3.0
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--heads", default="sensitive", choices=["sensitive", "all"])
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=1000)
    args = ap.parse_args()

    for d in sorted(glob.glob(os.path.join(args.run, "*"))):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta, image, recs = load(d)
        if len(recs) < 2:
            continue

        sc = min(1.0, args.max_side / max(image.size))
        thumb = image.resize((int(image.width * sc), int(image.height * sc)), Image.LANCZOS)
        W, H = thumb.size
        by = max(12, H // 26)
        bins = [by, max(12, int(by * W / H))]

        if args.heads == "sensitive":
            heads = sensitive_heads(recs, args.layer, args.n_heads)
        else:
            heads = np.arange(recs[0]["attn"].shape[1])

        n = len(recs)
        fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.2 * H / W + 1.2))
        axes = np.atleast_1d(axes)
        maps = [read_map(r, args.layer, heads, (H, W), bins) for r in recs]
        vmax = max(mm.max() for mm in maps)

        for ax, r, mm in zip(axes, recs, maps):
            ax.imshow(thumb)
            ax.imshow(mm, extent=[0, W, H, 0], cmap="jet", alpha=0.45,
                      vmin=0, vmax=vmax, interpolation="bilinear")
            ok = "correct" if r["q"]["correct"] else "wrong"
            ax.set_title("[%s] %s\npred=%s  gt=%s (%s)"
                         % (r["q"]["category"], r["q"]["question"][:64],
                            r["q"]["pred"][:18], r["q"]["gt"], ok), fontsize=9)
            ax.axis("off")

        fig.suptitle("%s  %dx%d   DAT layer idx %d, heads=%s %s"
                     % (meta["hash"], meta["size"][0], meta["size"][1], args.layer,
                        args.heads, list(np.array(heads)[:6])), fontsize=10)
        fig.tight_layout()
        out = os.path.join(d, "readmap_L%d_%s.png" % (args.layer, args.heads))
        fig.savefig(out, dpi=115, bbox_inches="tight")
        plt.close(fig)
        print("->", out, flush=True)


if __name__ == "__main__":
    main()
