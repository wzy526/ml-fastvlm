#!/usr/bin/env python3
"""Second diagnostic: what *does* move the sampling points?

Three displacement scales, all in normalized [-1,1] units and all comparable to
the reference-grid pitch:

  question   same image, different question   (question conditioning)
  image      different image, same layer      (image conditioning)
  offset     distance from the uniform reference grid (is it deformable at all?)

If `offset` is tiny the module has collapsed to a fixed grid pooling; if `image`
is large but `question` is tiny, sampling is image-driven only.
"""

import glob
import json
import os
import sys

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "/home/ea-cvfa-aigc-x2v-2/xzf/vis_samples/run_k0"

data = {}
for d in sorted(glob.glob(os.path.join(RUN, "*"))):
    if not os.path.exists(os.path.join(d, "meta.json")):
        continue
    meta = json.load(open(os.path.join(d, "meta.json")))
    locs = [np.load(os.path.join(d, "q%d.npz" % q["idx"]))["locs"]
            for q in meta["questions"]]
    data[meta["hash"]] = np.stack(locs)          # [Q, L, P, 2]

hashes = sorted(data)
L, P = data[hashes[0]].shape[1:3]
grps = 8
grid = int(round(np.sqrt(P / grps)))

# Reference grid, mirroring _grid_generate exactly: half-cell margin off the
# [-1,1] border, meshgrid(indexing='ij'), stacked as (x, y) -- note the channel
# order is x-first, and grid_sample is later fed locs[..., (1,0)].
m = 1.0 / max(grid - 1, 1)
ax = np.linspace(-1.0 + m, 1.0 - m, grid)
pitch = float(ax[1] - ax[0])
gy, gx = np.meshgrid(ax, ax, indexing="ij")
ref = np.stack([gx, gy], -1).reshape(-1, 2)                 # [grid*grid, 2]
ref = np.tile(ref, (grps, 1))                               # [P, 2]

print("grid=%d  groups=%d  points/layer=%d  pitch=%.4f" % (grid, grps, P, pitch))
print()


def rel(v):
    return "%.4f (%5.1f%% pitch)" % (v, 100 * v / pitch)


# 1. offset from the uniform reference grid
off = [np.linalg.norm(v - ref, axis=-1) for v in data.values()]
off = np.concatenate([o.reshape(-1) for o in off])
print("offset from uniform grid :", rel(off.mean()),
      " p50=%.3f p99=%.3f max=%.3f" % (np.percentile(off, 50),
                                       np.percentile(off, 99), off.max()))

# 2. question conditioning: same image, question pairs
qd = []
for h in hashes:
    v = data[h]
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            qd.append(np.linalg.norm(v[i] - v[j], axis=-1).reshape(-1))
qd = np.concatenate(qd)
print("same image, diff question:", rel(qd.mean()),
      " p99=%.3f max=%.3f" % (np.percentile(qd, 99), qd.max()))

# 3. image conditioning: different images, first question of each, layer-matched
idd = []
for a in range(len(hashes)):
    for b in range(a + 1, len(hashes)):
        idd.append(np.linalg.norm(data[hashes[a]][0] - data[hashes[b]][0],
                                  axis=-1).reshape(-1))
idd = np.concatenate(idd)
print("diff image,  first quest.:", rel(idd.mean()),
      " p99=%.3f max=%.3f" % (np.percentile(idd, 99), idd.max()))

print()
print("ratios:  image/question = %.1fx     offset/pitch = %.2f"
      % (idd.mean() / qd.mean(), off.mean() / pitch))

# per-layer view: which layers move at all
print()
print("per-layer mean displacement (offset | question | image):")
for li in range(L):
    o = np.mean([np.linalg.norm(v[:, li] - ref, axis=-1).mean() for v in data.values()])
    q = np.mean([np.linalg.norm(data[h][i, li] - data[h][j, li], axis=-1).mean()
                 for h in hashes for i in range(len(data[h]))
                 for j in range(i + 1, len(data[h]))])
    m = np.mean([np.linalg.norm(data[hashes[a]][0, li] - data[hashes[b]][0, li], axis=-1).mean()
                 for a in range(len(hashes)) for b in range(a + 1, len(hashes))])
    print("  L%d  %6.3f | %6.4f | %6.4f" % (li, o, q, m))
