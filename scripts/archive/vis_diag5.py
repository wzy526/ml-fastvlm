#!/usr/bin/env python3
"""Per-head, last-token attention response to the question.

An earlier pass averaged over 16 heads and 6 answer-slot tokens, either of which can
wash out a real signal. Here:

  * only the LAST answer query token is used -- the one that predicts the first
    answer token, so it is the read that actually decides the output;
  * every (layer, head) is scored separately, and the distribution over heads is
    reported, because a few specialized heads matter more than the average;
  * top-k overlap is reported next to centroid shift, since a multi-modal
    attention map can redistribute mass without moving its centroid.

The contrast baseline is a *different image* in the same question slot: that is
how far this attention map can travel at all.
"""

import glob
import json
import os
import sys

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ea-cvfa-aigc-x2v-2/xzf/vis_samples/run_k0_attn"
GRPS = 8

data = {}
for d in sorted(glob.glob(os.path.join(RUN, "*"))):
    if not os.path.exists(os.path.join(d, "meta.json")):
        continue
    meta = json.load(open(os.path.join(d, "meta.json")))
    recs = []
    for q in meta["questions"]:
        z = np.load(os.path.join(d, "q%d.npz" % q["idx"]))
        if "attn" not in z.files:
            continue
        a = z["attn"][:, :, -1, :]                       # [L, H, Ns] last token
        Ns = a.shape[-1]
        locs = z["locs"].reshape(z["locs"].shape[0], GRPS, Ns, 2).mean(axis=1)
        recs.append({"q": q, "a": a, "locs": locs})
    if len(recs) >= 2:
        data[meta["hash"]] = {"meta": meta, "recs": recs}

grid = int(round(np.sqrt(Ns)))
m = 1.0 / (grid - 1)
ax = np.linspace(-1 + m, 1 - m, grid)
pitch = float(ax[1] - ax[0])
print("last answer token only | grid=%d Ns=%d pitch=%.4f" % (grid, Ns, pitch))


def centroids(r):
    """attention-weighted read location per (layer, head) -> [L, H, 2]"""
    a = r["a"] / r["a"].sum(-1, keepdims=True)
    return np.einsum("lhs,lsc->lhc", a, r["locs"])


def topk_jac(a1, a2, k=10):
    """overlap of the top-k attended sampled points, per (layer, head)"""
    i1 = np.argsort(a1, -1)[..., -k:]
    i2 = np.argsort(a2, -1)[..., -k:]
    out = np.empty(i1.shape[:-1])
    for idx in np.ndindex(*i1.shape[:-1]):
        s1, s2 = set(i1[idx].tolist()), set(i2[idx].tolist())
        out[idx] = len(s1 & s2) / len(s1 | s2)
    return out


for h, D in data.items():
    recs, meta = D["recs"], D["meta"]
    W = meta["meta" if False else "size"][0]
    cents = [centroids(r) for r in recs]
    print("=" * 74)
    print("%s %s  Q=%d" % (h, meta["size"], len(recs)))

    shifts, jacs = [], []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            dv = np.linalg.norm(cents[i] - cents[j], axis=-1)       # [L, H]
            jc = topk_jac(recs[i]["a"], recs[j]["a"])
            shifts.append(dv)
            jacs.append(jc)
            print("  q%d vs q%d  centroid: mean=%.4f (%4.1f%% pitch)  "
                  "p90=%.4f  MAX=%.4f (%5.1f%% pitch, ~%4.0f px)   top10 overlap=%.2f"
                  % (i, j, dv.mean(), 100 * dv.mean() / pitch,
                     np.percentile(dv, 90), dv.max(), 100 * dv.max() / pitch,
                     dv.max() * 0.5 * W, jc.mean()))

    S = np.stack(shifts)
    print("  -> over all pairs: mean=%.4f  p99=%.4f  max=%.4f  |  top10 overlap mean=%.2f"
          % (S.mean(), np.percentile(S, 99), S.max(), np.stack(jacs).mean()))
    # which heads respond most
    per_head = S.mean(axis=(0, 1))
    order = np.argsort(per_head)[::-1][:4]
    print("     most question-sensitive heads: "
          + ", ".join("H%d:%.4f" % (k, per_head[k]) for k in order))

# baseline: how far does this map move for a DIFFERENT image?
hs = sorted(data)
if len(hs) >= 2:
    print("=" * 74)
    base = []
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            ca = centroids(data[hs[a]]["recs"][0])
            cb = centroids(data[hs[b]]["recs"][0])
            base.append(np.linalg.norm(ca - cb, axis=-1))
    B = np.stack(base)
    allq = np.stack([np.linalg.norm(centroids(D["recs"][i]) - centroids(D["recs"][j]), axis=-1)
                     for D in data.values() for i in range(len(D["recs"]))
                     for j in range(i + 1, len(D["recs"]))])
    print("read-centroid shift:  different QUESTION = %.4f (%.1f%% pitch)"
          % (allq.mean(), 100 * allq.mean() / pitch))
    print("                      different IMAGE    = %.4f (%.1f%% pitch)"
          % (B.mean(), 100 * B.mean() / pitch))
    print("                      question / image   = %.2f" % (allq.mean() / B.mean()))
