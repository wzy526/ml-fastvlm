#!/usr/bin/env python3
"""Focused render of the question-conditioned read.

The averaged heat map smears the signal twice over (attention mass duplicated
across all off_grps source positions, then averaged over heads). This version
keeps it sharp:

  top-k panel   a single head, the top-k attended keys only, drawn at their
                group-mean position with radius proportional to attention. The
                top-10 overlap between questions is ~0.5, so the marker sets
                should visibly differ.
  diff panel    attention of question i minus question 0 on the shared point
                set: red = this question looks here more, blue = less. This is
                the cleanest evidence that the read moves with the question.
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
    recs = []
    for q in meta["questions"]:
        z = np.load(os.path.join(d, "q%d.npz" % q["idx"]))
        if "attn" not in z.files:
            continue
        Ns = z["attn"].shape[-1]
        recs.append({
            "q": q,
            "attn": z["attn"][:, :, -1, :],
            "pos": z["locs"].reshape(z["locs"].shape[0], GRPS, Ns, 2).mean(1),  # [L,Ns,2]
        })
    return meta, image, recs


def best_head(recs, layer):
    cs = []
    for r in recs:
        a = r["attn"][layer] / r["attn"][layer].sum(-1, keepdims=True)
        cs.append(a @ r["pos"][layer])
    sh, n = np.zeros(cs[0].shape[0]), 0
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            sh += np.linalg.norm(cs[i] - cs[j], axis=-1)
            n += 1
    return int(np.argmax(sh / max(n, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=-1, help="-1 = most question-sensitive")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-side", type=int, default=1100)
    args = ap.parse_args()

    for d in sorted(glob.glob(os.path.join(args.run, "*"))):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta, image, recs = load(d)
        if len(recs) < 2:
            continue
        head = best_head(recs, args.layer) if args.head < 0 else args.head

        sc = min(1.0, args.max_side / max(image.size))
        thumb = image.resize((int(image.width * sc), int(image.height * sc)), Image.LANCZOS)
        W, H = thumb.size

        A = [r["attn"][args.layer][head] for r in recs]
        A = [a / a.sum() for a in A]
        P = [r["pos"][args.layer] for r in recs]

        n = len(recs)
        fig, axes = plt.subplots(2, n, figsize=(5.6 * n, 2 * 5.6 * H / W + 1.4))
        axes = np.atleast_2d(axes)

        for c, (r, a, p) in enumerate(zip(recs, A, P)):
            x = (p[:, 1] + 1) * 0.5 * W
            y = (p[:, 0] + 1) * 0.5 * H
            idx = np.argsort(a)[::-1][: args.topk]

            ax = axes[0, c]
            ax.imshow(thumb)
            s = a[idx] / a[idx].max()
            ax.scatter(x[idx], y[idx], s=40 + 900 * s, facecolors="none",
                       edgecolors="lime", linewidths=2.0, alpha=0.95)
            ax.scatter(x[idx], y[idx], s=8, color="yellow")
            ok = "correct" if r["q"]["correct"] else "wrong"
            ax.set_title("[%s] %s\npred=%s gt=%s (%s)"
                         % (r["q"]["category"], r["q"]["question"][:58],
                            r["q"]["pred"][:16], r["q"]["gt"], ok), fontsize=9)
            ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

            ax2 = axes[1, c]
            ax2.imshow(thumb.convert("L"), cmap="gray")
            dv = a - A[0]
            lim = np.abs(dv).max() or 1.0
            ax2.scatter(x, y, s=30 + 800 * np.abs(dv) / lim, c=dv, cmap="coolwarm",
                        vmin=-lim, vmax=lim, alpha=0.85, linewidths=0)
            ax2.set_title("attention change vs q0 (red = more)" if c else
                          "reference (q0)", fontsize=9)
            ax2.set_xlim(0, W); ax2.set_ylim(H, 0); ax2.axis("off")

        ov = []
        for i in range(n):
            for j in range(i + 1, n):
                s1 = set(np.argsort(A[i])[::-1][: args.topk].tolist())
                s2 = set(np.argsort(A[j])[::-1][: args.topk].tolist())
                ov.append(len(s1 & s2) / len(s1 | s2))
        fig.suptitle("%s  %dx%d   DAT layer idx %d, head %d   top-%d overlap=%.2f"
                     % (meta["hash"], meta["size"][0], meta["size"][1],
                        args.layer, head, args.topk, float(np.mean(ov))), fontsize=11)
        fig.tight_layout()
        out = os.path.join(d, "topk_L%d_H%d.png" % (args.layer, head))
        fig.savefig(out, dpi=115, bbox_inches="tight")
        plt.close(fig)
        print("-> %s  (head %d, top-%d overlap %.2f)"
              % (out, head, args.topk, float(np.mean(ov))), flush=True)


if __name__ == "__main__":
    main()
