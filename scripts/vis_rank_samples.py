#!/usr/bin/env python3
"""Rank multi-question samples by how strongly the read moves with the question.

For every image, sweeps all (layer, head) pairs and scores each by the top-k
overlap of attended keys between questions (lower = the model reads different
places) together with the read-centroid shift in pixels. Prints a table sorted
so the best Fig-1 candidates come first: all answers correct, low overlap.
"""

import argparse
import glob
import json
import os

import numpy as np

GRPS = 8


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    recs = []
    for q in meta["questions"]:
        z = np.load(os.path.join(d, "q%d.npz" % q["idx"]))
        if "attn" not in z.files:
            continue
        Ns = z["attn"].shape[-1]
        recs.append({
            "q": q,
            "attn": z["attn"][:, :, -1, :],                                   # [L,H,Ns]
            "pos": z["locs"].reshape(z["locs"].shape[0], GRPS, Ns, 2).mean(1),
        })
    return meta, recs


def score(recs, layer, head, topk):
    A = [r["attn"][layer][head] for r in recs]
    A = [a / a.sum() for a in A]
    P = recs[0]["pos"][layer]
    ov, sh = [], []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            s1 = set(np.argsort(A[i])[::-1][:topk].tolist())
            s2 = set(np.argsort(A[j])[::-1][:topk].tolist())
            ov.append(len(s1 & s2) / len(s1 | s2))
            sh.append(np.linalg.norm(A[i] @ P - A[j] @ P))
    return float(np.mean(ov)), float(np.mean(sh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-layer", type=int, default=2,
                    help="restrict to shallow DAT layers. w_hd falls from ~0.35 at "
                         "layer 0 to ~0.04 at layer 7, and the deep layers' offsets "
                         "saturate against the [-1,1] border, so their attention sits "
                         "on the image edge and is a clamping artifact, not a read.")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.run, "*"))):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta, recs = load(d)
        if len(recs) < 2:
            continue
        L, H = recs[0]["attn"].shape[:2]
        # Maximize the read shift, not minimize overlap: a head can have zero
        # top-k overlap while its centroid barely moves (different points, same
        # neighbourhood), which is invisible in a figure. Overlap is kept as a
        # reported tie-breaker.
        best = None
        for l in range(min(L, args.max_layer + 1)):
            for h in range(H):
                ov, sh = score(recs, l, h, args.topk)
                if best is None or sh > best[1]:
                    best = (ov, sh, l, h)
        ov, sh, l, h = best
        ncorr = sum(bool(r["q"]["correct"]) for r in recs)
        rows.append({
            "hash": meta["hash"], "size": meta["size"], "nq": len(recs),
            "ncorr": ncorr, "overlap": ov,
            "shift_px": sh * 0.5 * meta["size"][0], "layer": l, "head": h,
            "questions": [(r["q"]["category"], r["q"]["question"],
                           r["q"]["pred"], r["q"]["gt"], bool(r["q"]["correct"]))
                          for r in recs],
        })

    # all-correct first, then largest read shift
    rows.sort(key=lambda r: (-(r["ncorr"] == r["nq"]), -r["shift_px"]))

    print("%-9s %-12s %2s %2s %7s %8s %s" %
          ("hash", "size", "Q", "OK", "overlap", "shift_px", "layer/head"))
    print("-" * 72)
    for r in rows:
        print("%-9s %-12s %2d %2d %7.2f %8.0f  L%d/H%-2d %s"
              % (r["hash"], "%dx%d" % tuple(r["size"]), r["nq"], r["ncorr"],
                 r["overlap"], r["shift_px"], r["layer"], r["head"],
                 "<== all correct" if r["ncorr"] == r["nq"] else ""))

    print()
    print("top candidates in detail:")
    for r in rows[:5]:
        print("=" * 72)
        print("%s  %dx%d   Q=%d correct=%d   overlap=%.2f  shift=%.0f px  L%d/H%d"
              % (r["hash"], r["size"][0], r["size"][1], r["nq"], r["ncorr"],
                 r["overlap"], r["shift_px"], r["layer"], r["head"]))
        for cat, q, pred, gt, ok in r["questions"]:
            print("   [%-6s] %-72s pred=%-16s gt=%s %s"
                  % (cat, q[:72], pred[:16], gt, "OK" if ok else "x"))

    json.dump(rows, open(os.path.join(args.run, "ranking.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
