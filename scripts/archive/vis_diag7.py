#!/usr/bin/env python3
"""Is the learned offset content-adaptive, or a fixed re-drawing of the grid?

diag2 reports the offset magnitude, but a magnitude alone cannot tell a
deformable sampler from a static one: a module that shifts every point by the
same 0.05 regardless of input also scores "50% pitch offset". So split the
offset off = locs - ref into

  static   mean over all (image, question) samples  -> the part that is the
           same for every input, i.e. just a different fixed grid
  dynamic  per-sample residual                      -> the only part that can
           be called deformable

and then look at what the static part is:

  translation   one vector per layer      (whole grid slid sideways)
  per-group     one vector per (layer, g) (8 slid grids = a fixed multi-grid)
  structured    whatever is left           (spatially varying fixed warp)

Everything in normalized [-1,1] units and as % of the reference pitch.
"""

import glob
import json
import os
import sys

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "/home/ea-cvfa-aigc-x2v-2/xzf/vis_samples/acc_op10"
GRPS = 8

samples = []                                   # each [L, P, 2]
for d in sorted(glob.glob(os.path.join(RUN, "*"))):
    if not os.path.exists(os.path.join(d, "meta.json")):
        continue
    meta = json.load(open(os.path.join(d, "meta.json")))
    for q in meta["questions"]:
        samples.append(np.load(os.path.join(d, "q%d.npz" % q["idx"]))["locs"])
X = np.stack(samples).astype(np.float64)       # [S, L, P, 2]
S, L, P, _ = X.shape
Ns = P // GRPS
grid = int(round(np.sqrt(Ns)))

# reference grid exactly as in vis_diag2 / _grid_generate
m = 1.0 / max(grid - 1, 1)
ax = np.linspace(-1.0 + m, 1.0 - m, grid)
pitch = float(ax[1] - ax[0])
gy, gx = np.meshgrid(ax, ax, indexing="ij")
ref = np.tile(np.stack([gx, gy], -1).reshape(-1, 2), (GRPS, 1))   # [P, 2]

off = X - ref                                  # [S, L, P, 2]
static = off.mean(axis=0)                      # [L, P, 2]
dynamic = off - static                         # [S, L, P, 2]


def rms(v):
    """RMS vector length over every point."""
    return float(np.sqrt((v ** 2).sum(-1).mean()))


def pct(v):
    return "%.4f (%5.1f%% pitch)" % (v, 100 * v / pitch)


tot = rms(off)
sta = rms(static)
dyn = rms(dynamic)

print("samples=%d layers=%d points/layer=%d pitch=%.4f" % (S, L, P, pitch))
print()
print("offset RMS total   :", pct(tot))
print("  static  (fixed)  :", pct(sta), " -> %4.1f%% of energy" % (100 * sta ** 2 / tot ** 2))
print("  dynamic (input)  :", pct(dyn), " -> %4.1f%% of energy" % (100 * dyn ** 2 / tot ** 2))
print()

# decompose the static part
trans = static.mean(axis=1, keepdims=True)                          # [L, 1, 2]
sg = static.reshape(L, GRPS, Ns, 2)
gtrans = sg.mean(axis=2, keepdims=True)                             # [L, G, 1, 2]
struct = (sg - gtrans).reshape(L, P, 2)                             # residual warp

print("static part decomposed:")
print("  global translation      :", pct(rms(np.broadcast_to(trans, static.shape))))
print("  per-group translation   :", pct(rms(np.broadcast_to(gtrans, sg.shape).reshape(L, P, 2))),
      " (includes the global one)")
print("  structured warp residual:", pct(rms(struct)))
print()

# is the static warp spatially coherent (e.g. radial), or noise?  Correlate
# the static offset with the reference position: a pull toward / push away from
# the centre shows up as a strong (anti)correlation along each axis.
r = ref - ref.mean(0)
for name, k in (("x", 0), ("y", 1)):
    c = [np.corrcoef(static[l, :, k], r[:, k])[0, 1] for l in range(L)]
    print("  corr(static_%s, ref_%s) per layer: %s" % (name, name,
          " ".join("%+.2f" % v for v in c)))
print("    (+ = expansion away from centre, - = contraction toward centre)")
print()

print("per-layer RMS  (total | static | dynamic | dyn share):")
for l in range(L):
    t, s, d = rms(off[:, l]), rms(static[l]), rms(dynamic[:, l])
    print("  L%d  %.4f | %.4f | %.4f | %4.1f%%" % (l, t, s, d, 100 * d ** 2 / max(t ** 2, 1e-12)))
