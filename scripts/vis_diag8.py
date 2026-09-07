#!/usr/bin/env python3
"""Weight-level look at the offset-prediction chain.

Pipeline (weights shared across the 8 offset groups):

    img_lr -> conv_lr_dw (3x3 depthwise, kaiming) -> ln_1 -> silu
           -> conv_lr_proj (1x1, has bias)          -> avgpool 20x20
           -> x 2*sigmoid(proj_intention(intent))   -> ln_2 -> silu
           -> conv_off_proj (1x1, inter->2, ZERO init, no bias) -> (dx, dy)

Questions, in the order they can kill "deformable":

  1. conv_off_proj: are the dx and dy rows (anti)parallel?  |cos| ~ 1 means the
     module can only move points along one line.
  2. bias route: with a content-free input (h_hat = 0) the output is
     W_proj @ silu(beta_2).  If that reproduces the measured per-layer global
     translation, the constant offset is just LayerNorm's bias leaking through.
  3. input gain: J = W_proj @ diag(silu'(beta_2) * gamma_2) is the sensitivity to
     the normalized feature; ||J_row|| vs |t_pred| is a weight-only estimate of
     dynamic/static.
  4. conv_lr_dw: 3x3 kernel symmetry.  Odd (rotation-180) energy is what a
     derivative filter has; iid init expects center 1/9, odd 4/9, odd-x 3/9,
     odd-y 3/9.  A kernel that drifted to even/isotropic cannot signal
     direction; one that collapsed onto the center tap is a 1x1.
  5. cross-arm cosine of the early-chain weights.  HF seeds the DAT init the
     same way in every run, so cos ~ 1 between independently trained arms means
     those weights never moved -- the gradient reaching them through a
     near-zero conv_off_proj was negligible.

Usage:
  python vis_diag8.py name=ckpt_dir[:locs_run] [name=ckpt_dir[:locs_run] ...]
"""

import glob
import json
import os
import re
import sys

import numpy as np
from safetensors import safe_open

WANT = ('conv_lr_dw', 'ln_1', 'conv_lr_proj', 'proj_intention', 'ln_2', 'conv_off_proj')
GRPS = 8


def silu(x):
    return x / (1.0 + np.exp(-x))


def dsilu(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


def load_weights(ck):
    W = {}
    for f in sorted(glob.glob(os.path.join(ck, "*.safetensors"))):
        with safe_open(f, framework="np") as sf:
            for k in sf.keys():
                if 'self_attn' not in k or not any(w in k for w in WANT):
                    continue
                m = re.search(r'layers\.(\d+)\.', k)
                if not m:
                    continue
                name = k.split('self_attn.')[-1]
                W.setdefault(int(m.group(1)), {})[name] = sf.get_tensor(k).astype(np.float64)
    return W


def measured_translation(run):
    """Per-layer mean (locs - ref) over all samples/points, from a vis run."""
    S = []
    for d in sorted(glob.glob(os.path.join(run, "*"))):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        for q in json.load(open(os.path.join(d, "meta.json")))["questions"]:
            S.append(np.load(os.path.join(d, "q%d.npz" % q["idx"]))["locs"])
    X = np.stack(S).astype(np.float64)                    # [S, L, P, 2]
    P = X.shape[2]
    g = int(round(np.sqrt(P // GRPS)))
    m = 1.0 / max(g - 1, 1)
    ax = np.linspace(-1.0 + m, 1.0 - m, g)
    gy, gx = np.meshgrid(ax, ax, indexing="ij")
    ref = np.tile(np.stack([gx, gy], -1).reshape(-1, 2), (GRPS, 1))
    off = X - ref
    t = off.mean(axis=(0, 2))                             # [L, 2]
    dyn = off - off.mean(axis=0)                          # [S, L, P, 2]
    dyn_rms = np.sqrt((dyn ** 2).sum(-1).mean(axis=(0, 2)))   # [L]
    pitch = float(ax[1] - ax[0])
    return t, dyn_rms, pitch


def kernel_symmetry(K):
    """K: [C, 3, 3]. Energy fractions of center tap, odd-180, odd-x, odd-y."""
    e = (K ** 2).sum()
    center = (K[:, 1, 1] ** 2).sum() / e
    rot = K[:, ::-1, ::-1]
    odd180 = (((K - rot) / 2) ** 2).sum() / e
    oddx = (((K - K[:, :, ::-1]) / 2) ** 2).sum() / e
    oddy = (((K - K[:, ::-1, :]) / 2) ** 2).sum() / e
    # do all channels share one template?
    kn = K.reshape(len(K), -1)
    kn = kn / (np.linalg.norm(kn, axis=1, keepdims=True) + 1e-12)
    mean = kn.mean(0)
    coh = float(np.linalg.norm(mean))          # 0 = random directions, 1 = identical
    return center, odd180, oddx, oddy, coh


def main():
    arms = []
    for a in sys.argv[1:]:
        name, rest = a.split('=', 1)
        ck, _, run = rest.partition(':')
        arms.append((name, ck, run or None))
    if not arms:
        sys.exit(__doc__)

    allW = {}
    for name, ck, run in arms:
        W = load_weights(ck)
        allW[name] = W
        layers = sorted(W)
        meas = measured_translation(run) if run else None

        print("=" * 78)
        print("ARM %s   layers=%s" % (name, layers))
        print("=" * 78)

        print("\n[1] conv_off_proj rows (dx, dy): norm, cosine   |   ln_2 gamma/beta rms")
        for L in layers:
            p = W[L]['conv_off_proj.weight'].reshape(2, -1)
            g2, b2 = W[L]['ln_2.weight'], W[L]['ln_2.bias']
            cos = float(p[0] @ p[1] / (np.linalg.norm(p[0]) * np.linalg.norm(p[1]) + 1e-12))
            print("  L%-2d |w_dx|=%.4f |w_dy|=%.4f cos=%+.3f   |  rms(g2)=%.3f rms(b2)=%.3f  rms(b2)/rms(g2)=%.2f"
                  % (L, np.linalg.norm(p[0]), np.linalg.norm(p[1]), cos,
                     np.sqrt((g2 ** 2).mean()), np.sqrt((b2 ** 2).mean()),
                     np.sqrt((b2 ** 2).mean()) / (np.sqrt((g2 ** 2).mean()) + 1e-12)))

        print("\n[2] bias route t_pred = W_proj @ silu(beta_2)  vs  measured global translation")
        print("    and [3] input gain ||J_row|| (J = W_proj @ diag(silu'(b2)*g2))  vs measured dynamic RMS")
        for i, L in enumerate(layers):
            p = W[L]['conv_off_proj.weight'].reshape(2, -1)
            g2, b2 = W[L]['ln_2.weight'], W[L]['ln_2.bias']
            t_pred = p @ silu(b2)
            J = p * (dsilu(b2) * g2)[None, :]
            gain = np.sqrt((J ** 2).sum(1))                  # per row
            line = "  L%-2d t_pred=(%+.4f,%+.4f) |t_pred|=%.4f" % (L, t_pred[0], t_pred[1], np.linalg.norm(t_pred))
            if meas is not None:
                t, dyn_rms, pitch = meas
                tm = t[i]
                cos = float(t_pred @ tm / (np.linalg.norm(t_pred) * np.linalg.norm(tm) + 1e-12))
                line += "  meas=(%+.4f,%+.4f) |t|=%.4f  cos=%+.2f ratio=%.2f" % (
                    tm[0], tm[1], np.linalg.norm(tm), cos, np.linalg.norm(t_pred) / (np.linalg.norm(tm) + 1e-12))
            line += "   gain=(%.4f,%.4f)" % (gain[0], gain[1])
            if meas is not None:
                line += " meas_dyn=%.4f" % dyn_rms[i]
            print(line)

        print("\n[4] conv_lr_dw 3x3 kernel symmetry (energy fractions; iid init: center .111 odd180 .444 oddx .333 oddy .333)")
        for L in layers:
            K = W[L]['conv_lr_dw.weight'][:, 0]
            c, o, ox, oy, coh = kernel_symmetry(K)
            print("  L%-2d center=%.3f odd180=%.3f oddx=%.3f oddy=%.3f  channel-coherence=%.3f  rms=%.4f"
                  % (L, c, o, ox, oy, coh, np.sqrt((K ** 2).mean())))

        print("\n[5] other constant routes: conv_lr_proj bias vs weight, ln_1 beta, proj_intention bias (gate at zero intent)")
        for L in layers:
            lp_w = W[L]['conv_lr_proj.weight'].reshape(W[L]['conv_lr_proj.weight'].shape[0], -1)
            lp_b = W[L].get('conv_lr_proj.bias')
            b1 = W[L]['ln_1.bias']; g1 = W[L]['ln_1.weight']
            pi_w = W[L].get('proj_intention.weight'); pi_b = W[L].get('proj_intention.bias')
            s = "  L%-2d rms(lr_proj.w)=%.4f" % (L, np.sqrt((lp_w ** 2).mean()))
            if lp_b is not None:
                s += " rms(lr_proj.b)=%.4f" % np.sqrt((lp_b ** 2).mean())
            s += "  rms(g1)=%.3f rms(b1)=%.3f" % (np.sqrt((g1 ** 2).mean()), np.sqrt((b1 ** 2).mean()))
            if pi_w is not None:
                s += "  rms(int.w)=%.4f" % np.sqrt((pi_w ** 2).mean())
                if pi_b is not None:
                    gate0 = 2.0 / (1.0 + np.exp(-pi_b))
                    s += " gate@0: mean=%.3f std=%.3f" % (gate0.mean(), gate0.std())
            print(s)
        print()

    if len(arms) > 1:
        print("=" * 78)
        print("CROSS-ARM cosine of flattened weights per layer (~1 => same init, never moved)")
        print("=" * 78)
        names = [a[0] for a in arms]
        for key in ('conv_lr_dw.weight', 'conv_lr_proj.weight', 'proj_intention.weight',
                    'ln_1.weight', 'ln_2.weight', 'ln_2.bias', 'conv_off_proj.weight'):
            print("\n  %s" % key)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    Wa, Wb = allW[names[i]], allW[names[j]]
                    cs = []
                    for L in sorted(set(Wa) & set(Wb)):
                        if key not in Wa[L] or key not in Wb[L]:
                            continue
                        a, b = Wa[L][key].ravel(), Wb[L][key].ravel()
                        cs.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    if cs:
                        print("    %s vs %s : %s" % (names[i], names[j], " ".join("%+.3f" % c for c in cs)))


if __name__ == "__main__":
    main()
