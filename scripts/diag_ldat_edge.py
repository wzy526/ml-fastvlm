#!/usr/bin/env python
"""How much density the final Q-LAS head puts on the window border.

For each HR-Bench question (first option rotation) this runs LDatRunner.prepare
with the main-table settings and compares, per border (left / right / top /
bottom column or row of the window's native ViT grid):
  prior  the where-head density softmax(log p) alone
  final  softmax(log p + head residual), the density the sampler uses
  resid  mean head residual on that border minus its interior mean (logits)
  tokens share of the 20 mosaic columns / rows inside the outer 5% of the window
A ratio of 1 is what a flat density gives.

    python scripts/diag_ldat_edge.py --model_path $E/checkpoint-1000 \
        --where_head_pt $E/loc/where_lm8_ce20.pt --ldat_pt $X/hinge_noratio_ldat.pt --n 100
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.expanduser(os.environ.get("LMMS_EVAL_DIR", "~/lmms-eval-oldvstar")))

from ldat_eval.infer import LDatRunner  # noqa: E402
from offset_feat_train import inv_cdf  # noqa: E402
from vis_hrbench_sampling import _dataset  # noqa: E402

SIDES = ("left", "right", "top", "bottom")


def border_stats(q, lp):
    prior = torch.softmax(lp.flatten(), 0).reshape(lp.shape)
    res = (q.clamp_min(1e-12).log() - lp)
    out = {}
    for name, d in (("prior", prior), ("final", q)):
        col, row = d.sum(0), d.sum(1)
        out[name] = [float(col[0] / col.mean()), float(col[-1] / col.mean()),
                     float(row[0] / row.mean()), float(row[-1] / row.mean())]
    inner = res[1:-1, 1:-1].mean()
    out["resid"] = [float(res[:, 0].mean() - inner), float(res[:, -1].mean() - inner),
                    float(res[0, :].mean() - inner), float(res[-1, :].mean() - inner)]
    return out


def edge_tokens(d, n, frac=0.05):
    c = inv_cdf(d, (torch.arange(n, device=d.device) + 0.5) / n)
    return float((c < frac).float().mean()), float((c > 1 - frac).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--where_head_pt", required=True)
    ap.add_argument("--ldat_pt", required=True)
    ap.add_argument("--split", default="hrbench_8k")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds, prompt_of, visual = _dataset(args.split)
    seen, idx = set(), []
    for i in range(len(ds)):
        d = ds[i]
        key = (d["category"], d["question"].strip())
        if key not in seen:
            seen.add(key)
            idx.append(i)
    idx = idx[:args.n]
    runner = LDatRunner(args.model_path, args.where_head_pt, args.ldat_pt, args.device,
                        crop_tok=400, lr_tok=640, dat_alpha=0.8, dat_ratio=0.0, dat_k=0.6,
                        src_scale=4.0, where_layer=8, fast_prefix=True)
    acc = {k: [] for k in ("prior", "final", "resid")}
    tok = []
    for k, i in enumerate(idx):
        d = ds[i]
        with torch.inference_mode():
            pr = runner.prepare(visual(d)[0], prompt_of(d))
        if pr is None:
            continue
        w = pr["wins"][0]
        s = border_stats(w["q"].float(), w["lp"].float())
        for key in acc:
            acc[key].append(s[key])
        lx = edge_tokens(w["dx"], w["gw"])
        ly = edge_tokens(w["dy"], w["gh"])
        tok.append([lx[0], lx[1], ly[0], ly[1]])
        if (k + 1) % 20 == 0:
            print(f"[edge] {k + 1}/{len(idx)}", flush=True)
    n = len(tok)
    print(f"\n[edge] {n} windows, {args.split}. Ratios: border column / row mass over the mean one (1 = flat)")
    print(f"{'':8s}" + "".join(f"{s:>10s}" for s in SIDES))
    for key in ("prior", "final"):
        a = np.array(acc[key])
        print(f"{key:8s}" + "".join(f"{v:10.2f}" for v in a.mean(0)))
    a = np.array(acc["resid"])
    print(f"{'resid':8s}" + "".join(f"{v:+10.2f}" for v in a.mean(0)) + "   (logits, border minus interior)")
    t = np.array(tok)
    print(f"{'tokens':8s}" + "".join(f"{v:10.3f}" for v in t.mean(0)) + "   (share in the outer 5%; flat ~0.05)")
    a = np.array(acc["final"])
    print(f"\n[edge] windows whose final left column is >2x the mean: {int((a[:, 0] > 2).sum())}/{n}, "
          f"right: {int((a[:, 1] > 2).sum())}/{n}, top: {int((a[:, 2] > 2).sum())}/{n}, bottom: {int((a[:, 3] > 2).sum())}/{n}")


if __name__ == "__main__":
    main()
