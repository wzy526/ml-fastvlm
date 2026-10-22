#!/usr/bin/env python3
"""Are the sampling points saturating against the [-1,1] clamp?

_sample_hd_from_off_guide applies a straight-through clamp:
    sample_locs = x + (x.clamp(-1,1) - x).detach()
so any offset that overshoots is pinned to the border. Measured offsets average
703% of the grid pitch and reach 1.5 in some layers, which suggests widespread
saturation -- and saturated points sit on the image edge, where the top-attended
keys were observed to cluster.

Reports the fraction of pinned coordinates per layer, and how much of the
attention mass lands on pinned points.
"""

import glob
import json
import os
import sys

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ea-cvfa-aigc-x2v-2/xzf/vis_samples/run_all19"
GRPS = 8
EPS = 1e-3

tot = {"pin_y": [], "pin_x": [], "pin_any": [], "mass": [], "top12_pin": []}
per_layer = {}

for d in sorted(glob.glob(os.path.join(RUN, "*"))):
    if not os.path.exists(os.path.join(d, "meta.json")):
        continue
    meta = json.load(open(os.path.join(d, "meta.json")))
    for q in meta["questions"]:
        z = np.load(os.path.join(d, "q%d.npz" % q["idx"]))
        locs = z["locs"]                                    # [L, P, 2]
        L, P, _ = locs.shape
        Ns = P // GRPS
        pin = np.abs(locs) > 1.0 - EPS                      # [L, P, 2]
        tot["pin_y"].append(pin[..., 0].mean())
        tot["pin_x"].append(pin[..., 1].mean())
        tot["pin_any"].append(pin.any(-1).mean())
        for li in range(L):
            per_layer.setdefault(li, []).append(pin[li].any(-1).mean())

        if "attn" in z.files:
            a = z["attn"][:, :, -1, :]                      # [L, H, Ns]
            a = a / a.sum(-1, keepdims=True)
            # a key is "pinned" if any of its grps source points is pinned
            kp = pin.any(-1).reshape(L, GRPS, Ns).any(1)    # [L, Ns]
            tot["mass"].append(float((a * kp[:, None, :]).sum(-1).mean()))
            idx = np.argsort(a, -1)[..., -12:]
            tot["top12_pin"].append(float(np.take_along_axis(
                np.broadcast_to(kp[:, None, :], a.shape), idx, -1).mean()))

print("sampling points pinned to the [-1,1] border (|coord| > 1-1e-3):")
print("  y coordinate : %.1f%%" % (100 * np.mean(tot["pin_y"])))
print("  x coordinate : %.1f%%" % (100 * np.mean(tot["pin_x"])))
print("  either       : %.1f%%" % (100 * np.mean(tot["pin_any"])))
print()
print("per DAT layer (fraction of points pinned):")
for li in sorted(per_layer):
    print("  L%d  %5.1f%%" % (li, 100 * np.mean(per_layer[li])))
print()
if tot["mass"]:
    print("attention mass landing on pinned keys : %.1f%%" % (100 * np.mean(tot["mass"])))
    print("of the top-12 attended keys, pinned   : %.1f%%" % (100 * np.mean(tot["top12_pin"])))
