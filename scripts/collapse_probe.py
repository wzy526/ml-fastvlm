#!/usr/bin/env python3
"""Zero-training measurements behind the two "collapse" arguments (1022).

Setting: V* Bench, a DAT checkpoint, LR --tok_budget tokens, the HD side on
the real image. Per sample six answers and a set of statistics:

  off             no HD
  oracle          400 points forced onto the GT window           (K/V bypass)
  oracle_rand     same-size window elsewhere (no overlap)
  oracle_mean_v   oracle grid, value_hd replaced by its mean over the 400
                  points (keys untouched)      -> LM only ever sees beta * v_bar
  oracle_mean_kv  keys AND values replaced by their means (HD attention
                  uniform, beta from one key)
  real            the checkpoint's learned offsets

Collapse 1 (readout: 400 keys -> one scalar beta + one vector v_bar)
  * acc(oracle_mean_v) ~ acc(oracle): the answer never used the per-point
    content; it used beta and the window mean.
  * per layer: spread = mean_j ||v_j - v_bar|| / ||v_bar|| and the
    participation-ratio rank of {v_j - v_bar} (400 x kv_dim) -- how much
    content there is to lose in the first place;
  * cos(v_bar_oracle, v_bar_rand): is the HD vector even window-specific;
  * beta (HD attention mass at the merge, mean over answer queries / heads /
    DAT layers) per sample; acc gain vs beta quartile (saturation, not slope).

Collapse 2 (sampler: where learned through read, gradient vanishes)
  --grad: teacher-forced LM loss on the GT letter under the learned grid;
  ||dL_LM/dp_j|| per sampling point vs the direct box-pull
  ||dL_box/dp_j|| (huber to the oracle grid, delta 0.1, the off_sup target),
  and the same ratio on the offset parameters (conv_off_proj.weight) --
  what the coupled sampler receives vs what a decoupled supervision gives;
  direction: cos(dL_LM/dp_j, p_j - p_j*) per point (40-bin histogram per
  sample), within-grid coherence; --dump_grad N writes per-point grids and
  gradients of N samples per shard for the quiver figure (collapse_fig.py).

Shard with --shard/--shards; --summarize DIR merges the shard jsons.

  E=~/vldat_experiments/0924_miniE1_2k
  for i in 0..7: CUDA_VISIBLE_DEVICES=$i python scripts/collapse_probe.py \
      --model_path $E/checkpoint-1000 --tok_budget 640 --grad --dump_grad 2 \
      --shard $i --shards 8 --out $E/loc/collapse/s$i.json
  python scripts/collapse_probe.py --summarize $E/loc/collapse
  python scripts/collapse_fig.py --dir $E/loc/collapse --out fig_collapse2.pdf
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_whd_qwen35 import (  # noqa: E402
    STATE, TOK_PX, build_inputs, extract_letter, load_vstar, oracle_locs, set_force_locs,
)

CFGS = ["off", "oracle", "oracle_rand", "oracle_mean_v", "oracle_mean_kv", "real"]
REC = {"mode": None, "stats": None, "beta": None, "grids": None, "grid_layers": None, "cur_layer": None}
COS_BINS = 40                                   # histogram of cos(g_LM, p - p*) over [-1, 1]


def install_hooks():
    from llava.model.language_model.modeling_qwen3_5_dat import Qwen3_5AttentionDAT as A

    orig_sample = A._sample_hd_from_off_guide

    def sample_patched(self, *a, **kw):
        REC["cur_layer"] = self.layer_idx
        key_hd, value_hd, locs = orig_sample(self, *a, **kw)
        li = self.layer_idx
        if REC["stats"] is not None and li not in REC["stats"]:
            with torch.no_grad():
                V = value_hd[-1].float()                    # [Ns, kv_dim], answer slot
                vbar = V.mean(0)
                D = V - vbar
                spread = float(D.norm(dim=1).mean() / vbar.norm().clamp(min=1e-6))
                s = torch.linalg.svdvals(D)
                pr = float((s ** 2).sum() ** 2 / (s ** 4).sum().clamp(min=1e-12))
                REC["stats"][li] = {"spread": spread, "pr": pr, "vbar": vbar.cpu()}
        if REC["mode"] in ("mean_v", "mean_kv"):
            value_hd = value_hd.mean(dim=1, keepdim=True).expand_as(value_hd).contiguous()
            if REC["mode"] == "mean_kv":
                key_hd = key_hd.mean(dim=1, keepdim=True).expand_as(key_hd).contiguous()
        return key_hd, value_hd, locs

    A._sample_hd_from_off_guide = sample_patched

    orig_merge = A._merge_two_pass_lse

    def merge_patched(self, out1, lse1, out2, lse2, ans_start, ans_end):
        if REC["beta"] is not None and self.layer_idx not in REC["beta"]:
            with torch.no_grad():
                l1 = lse1[:, :, ans_start:ans_end].float()
                l2 = lse2.float()
                if self.hd_gate is not None:
                    l2 = l2 + F.logsigmoid(self.hd_gate)
                REC["beta"][self.layer_idx] = float((l2 - torch.logaddexp(l1, l2)).exp().mean())
        return orig_merge(self, out1, lse1, out2, lse2, ans_start, ans_end)

    A._merge_two_pass_lse = merge_patched

    orig_gs = F.grid_sample

    def gs_patched(inp, grid, *a, **kw):
        if REC["grids"] is not None and grid.requires_grad:
            REC["grids"].append(grid)
            REC["grid_layers"].append(REC["cur_layer"])
        return orig_gs(inp, grid, *a, **kw)

    F.grid_sample = gs_patched


def run_cfg(model, processor, base, hd_extra, locs, mode, gen_kwargs, want_stats=False):
    REC["mode"] = mode
    REC["stats"] = {} if want_stats else None
    REC["beta"] = {}
    set_force_locs(model, locs)
    inputs = dict(base)
    if hd_extra is not None:
        inputs.update(hd_extra)
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    n_in = inputs["input_ids"].shape[1]
    text = processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
    beta = REC["beta"]
    stats = REC["stats"]
    REC["mode"] = None
    REC["stats"] = None
    REC["beta"] = None
    return text, (float(np.mean(list(beta.values()))) if beta else None), stats


def grad_pass(model, processor, base, hd_extra, target, gt_letter, off_params, dump=None):
    """Learned grid: ||dL_LM/dp|| vs ||dL_box/dp|| per point and on the offset params.
    dump: path prefix -> <prefix>.npz with the per-layer grids / LM gradients
    (for the quiver figure, scripts/collapse_fig.py)."""
    target_grid, target_box = target
    set_force_locs(model, None)
    tok = processor.tokenizer
    tid = tok.encode(gt_letter, add_special_tokens=False)
    if not tid:
        return None
    tid = tid[0]
    REC["grids"], REC["grid_layers"] = [], []
    inputs = dict(base)
    inputs.update(hd_extra)
    for p in off_params:
        p.grad = None
    with torch.enable_grad():
        out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1].float()
        loss_lm = F.cross_entropy(logits[None], torch.tensor([tid], device=logits.device))
        grids, grid_layers = REC["grids"], REC["grid_layers"]
        REC["grids"] = REC["grid_layers"] = None
        if not grids:
            return None
        g_lm_pts = torch.autograd.grad(loss_lm, grids, retain_graph=True, allow_unused=True)
        g_lm_par = torch.autograd.grad(loss_lm, off_params, retain_graph=True, allow_unused=True)
        tgt = target_grid.to(grids[0].device)                      # [Ns, 2]
        delta = 0.1
        loss_box, g_box_pts = 0.0, []
        for g in grids:
            lg = g.shape[0]
            pts = g.reshape(lg, -1, 2)                              # [(l g), Ns, 2]
            diff = pts - tgt[None]
            loss_box = loss_box + F.huber_loss(pts, tgt[None].expand_as(pts), delta=delta,
                                               reduction="sum") / (pts.numel() // 2)
            g_box_pts.append((diff.clamp(-delta, delta) / (pts.numel() // 2)).reshape_as(g))
        g_box_par = torch.autograd.grad(loss_box, off_params, allow_unused=True)

    def pt_norms(gs):
        return torch.cat([g.detach().float().reshape(-1, 2).norm(dim=1) for g in gs if g is not None])

    def par_norm(gs):
        return float(torch.sqrt(sum((g.detach().float() ** 2).sum() for g in gs if g is not None)
                                + torch.tensor(0.0, device=grids[0].device)))

    n_lm, n_box = pt_norms(g_lm_pts), pt_norms(g_box_pts)
    # Direction, not magnitude: does the LM gradient push the sampling points
    # toward the oracle grid?  cos(g_lm, p - p*) > 0 means descent moves the
    # point toward the target (noise hypothesis: mean ~0, half positive).
    # Coherence: |mean_j g_j| / mean_j |g_j| over the Ns points of one grid
    # (1 = all points pushed the same way, ~1/sqrt(Ns) = independent noise).
    cos_l, coh_l, cos_layer, dump_arrays = [], [], {}, {}
    for gi_, g in enumerate(g_lm_pts):
        if g is None:
            continue
        gp = g.detach().float().reshape(g.shape[0], -1, 2)      # [(l g), Ns, 2]
        pts = grids[gi_].detach().float().reshape(gp.shape)
        diff = pts - tgt[None].float()
        far = diff.norm(dim=-1) > 1e-3
        c = F.cosine_similarity(gp, diff, dim=-1)[far]
        cos_l.append(c)
        coh_l.append(gp.mean(dim=1).norm(dim=-1) / gp.norm(dim=-1).mean(dim=1).clamp(min=1e-12))
        li = grid_layers[gi_]
        cos_layer[str(li)] = float(c.mean()) if c.numel() else float("nan")
        if dump is not None:
            dump_arrays[f"grid_{li}"] = pts.cpu().numpy().astype(np.float16)
            dump_arrays[f"glm_{li}"] = gp.cpu().numpy().astype(np.float32)
    cos_all = torch.cat(cos_l) if cos_l else torch.zeros(1)
    coh_all = torch.cat(coh_l) if coh_l else torch.zeros(1)
    hist = torch.histc(cos_all.clamp(-1, 1), bins=COS_BINS, min=-1, max=1)
    if dump is not None:
        os.makedirs(os.path.dirname(os.path.abspath(dump)), exist_ok=True)
        np.savez_compressed(dump + ".npz", target_grid=target_grid.cpu().numpy(),
                            target_box=target_box.cpu().numpy(),
                            layers=np.array(sorted({grid_layers[k] for k in range(len(grids)) if g_lm_pts[k] is not None})),
                            **dump_arrays)
    return {
        "loss_lm": float(loss_lm),
        "pt_lm_median": float(n_lm.median()), "pt_lm_p90": float(n_lm.quantile(0.9)),
        "pt_box_median": float(n_box.median()),
        "par_lm": par_norm(g_lm_par), "par_box": par_norm(g_box_par),
        "cos_mean": float(cos_all.mean()), "cos_pos_frac": float((cos_all > 0).float().mean()),
        "coh_mean": float(coh_all.mean()), "ns": int(tgt.shape[0]),
        "cos_hist": [int(x) for x in hist.tolist()], "cos_layer": cos_layer,
    }


def summarize(rows):
    rows = [r for r in rows if r.get("has_box")]
    n = len(rows)
    print(f"[collapse] {n} samples with a GT box")
    acc = {c: sum(r["ok"][c] for r in rows) / max(n, 1) * 100 for c in CFGS}
    print("acc  " + "  ".join(f"{c}={acc[c]:.1f}" for c in CFGS))
    same = sum(r["pred"]["oracle"] == r["pred"]["oracle_mean_v"] for r in rows)
    print(f"pred(oracle) == pred(oracle_mean_v): {same}/{n};  "
          f"== pred(oracle_mean_kv): {sum(r['pred']['oracle'] == r['pred']['oracle_mean_kv'] for r in rows)}/{n}")
    layers = sorted({int(k) for r in rows for k in r["layer"]})
    print("layer   spread   PR-rank   cos(vbar_oracle, vbar_rand)")
    for li in layers:
        sp = [r["layer"][str(li)]["spread"] for r in rows if str(li) in r["layer"]]
        pr = [r["layer"][str(li)]["pr"] for r in rows if str(li) in r["layer"]]
        cs = [r["layer"][str(li)]["cos_rand"] for r in rows if str(li) in r["layer"]]
        print(f"  {li:3d}   {np.mean(sp):.3f}   {np.mean(pr):6.1f}    {np.nanmean(cs):.3f}")
    for cfg in ("oracle", "real"):
        b = np.array([r["beta"][cfg] for r in rows])
        gain = np.array([int(r["ok"][cfg]) - int(r["ok"]["off"]) for r in rows])
        qs = np.quantile(b, [0.25, 0.5, 0.75])
        bins = np.digitize(b, qs)
        print(f"beta({cfg}) mean {b.mean():.3f}; gain vs off by beta quartile: " +
              "  ".join(f"Q{k + 1}[{b[bins == k].mean():.2f}] {gain[bins == k].mean() * 100:+.1f} (n={int((bins == k).sum())})"
                        for k in range(4)))
    g = [r["grad"] for r in rows if r.get("grad")]
    if g:
        r_pt = np.array([x["pt_lm_median"] / max(x["pt_box_median"], 1e-12) for x in g])
        r_par = np.array([x["par_lm"] / max(x["par_box"], 1e-12) for x in g])
        print(f"grad (n={len(g)}): per-point ||dL_LM/dp|| median {np.median([x['pt_lm_median'] for x in g]):.2e} "
              f"vs ||dL_box/dp|| {np.median([x['pt_box_median'] for x in g]):.2e}  -> ratio median {np.median(r_pt):.3g}")
        print(f"            on conv_off_proj: ||g_LM|| / ||g_box|| median {np.median(r_par):.3g} "
              f"(q25 {np.quantile(r_par, .25):.3g}, q75 {np.quantile(r_par, .75):.3g})")
        if "cos_mean" in g[0]:
            ns = g[0].get("ns", 0)
            print(f"            direction: cos(g_LM, p - p*) mean {np.mean([x['cos_mean'] for x in g]):+.3f} "
                  f"(toward-target fraction {np.mean([x['cos_pos_frac'] for x in g]):.3f}; noise -> 0 / 0.5); "
                  f"coherence |mean g|/mean|g| {np.mean([x['coh_mean'] for x in g]):.3f} "
                  f"(noise -> {1 / math.sqrt(max(ns, 1)):.3f}, Ns={ns})")
        if "cos_layer" in g[0]:
            lay = sorted({int(k) for x in g for k in x["cos_layer"]})
            print("            cos by layer: " + "  ".join(
                f"L{li} {np.nanmean([x['cos_layer'].get(str(li), np.nan) for x in g]):+.3f}" for li in lay))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path")
    ap.add_argument("--vstar_root", default="~/data/vstar_bench")
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--tok_budget", type=int, default=640)
    ap.add_argument("--min_pixels", type=int, default=1024)
    ap.add_argument("--hd_cap", type=int, default=5017600)
    ap.add_argument("--hr_scale", type=int, default=3)
    ap.add_argument("--oracle_min_cells", type=int, default=20)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--dump_grad", type=int, default=0,
                    help="with --grad: save per-point grids / LM gradients (+ a 1024px image copy) "
                         "for the first N boxed samples of this shard, for the quiver figure")
    ap.add_argument("--dump_dir", default=None, help="default: <dir of --out>/grad_dump")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summarize", default=None, help="dir of shard jsons -> merged summary")
    args = ap.parse_args()

    if args.summarize:
        rows = []
        for fn in sorted(glob.glob(os.path.join(os.path.expanduser(args.summarize), "*.json"))):
            rows += json.load(open(fn))
        summarize(rows)
        return
    if not args.model_path:
        raise SystemExit("--model_path required")

    from transformers import AutoProcessor
    from llava.model.language_model.modeling_qwen3_5_dat import (
        Qwen3_5AttentionDAT, Qwen3_5DATForConditionalGeneration)

    install_hooks()
    model = Qwen3_5DATForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation=args.attn).eval()
    model.requires_grad_(False)
    off_params = []
    for m in model.modules():
        if isinstance(m, Qwen3_5AttentionDAT):
            for sub in (m.conv_off_proj,):
                for p in sub.parameters():
                    p.requires_grad_(True)
                    off_params.append(p)
    args.grid_size = int((getattr(model.config, "dat_extra_args", None) or {}).get("grid_size", 20))
    args.hd_source = "oracle"
    args.multi_box = "union"
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels, max_pixels=args.tok_budget * TOK_PX)
    hr_processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=TOK_PX, max_pixels=100_000_000)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                      temperature=None, top_p=None, top_k=None)

    samples = load_vstar(args)
    mine = [(i, s) for i, s in enumerate(samples) if i % args.shards == args.shard]
    print(f"[collapse] shard {args.shard}/{args.shards}: {len(mine)} samples, grid {args.grid_size}, "
          f"tok {args.tok_budget}, grad={args.grad}", flush=True)
    dump_dir = args.dump_dir or (os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(args.out))),
                                             "grad_dump") if args.out else "grad_dump")
    n_dumped = 0
    rows = []
    for gi, s in mine:
        STATE["sample_idx"] = gi
        base, hd_extra, locs = build_inputs(s, gi, samples, processor, hr_processor, args, device, dtype)
        row = {"i": gi, "gt": s["gt"], "cat": s["category"], "has_box": bool(s["bboxes"]),
               "pred": {}, "ok": {}, "beta": {}, "layer": {}}
        if not s["bboxes"]:
            rows.append(row)
            continue
        lr_thw, hd_w, hd_h = STATE["geom"]
        rand_locs = oracle_locs(s, gi, hd_w, hd_h, args.grid_size, args.oracle_min_cells, "oracle_rand")
        plan = [("off", None, None, None), ("oracle", hd_extra, locs, None),
                ("oracle_rand", hd_extra, rand_locs, None),
                ("oracle_mean_v", hd_extra, locs, "mean_v"), ("oracle_mean_kv", hd_extra, locs, "mean_kv"),
                ("real", hd_extra, None, None)]
        vbars = {}
        for name, hd, lc, mode in plan:
            text, beta, stats = run_cfg(model, processor, base, hd, lc, mode, gen_kwargs,
                                        want_stats=name in ("oracle", "oracle_rand"))
            pred = extract_letter(text)
            row["pred"][name] = pred
            row["ok"][name] = bool(pred == s["gt"])
            row["beta"][name] = beta
            if stats:
                vbars[name] = stats
        for li, st in vbars.get("oracle", {}).items():
            vr = vbars.get("oracle_rand", {}).get(li)
            cos = float(F.cosine_similarity(st["vbar"], vr["vbar"], dim=0)) if vr else float("nan")
            row["layer"][str(li)] = {"spread": st["spread"], "pr": st["pr"], "cos_rand": cos}
        if args.grad and STATE.get("target") is not None:
            dump = None
            if n_dumped < args.dump_grad:
                dump = os.path.join(dump_dir, f"{gi:04d}")
                os.makedirs(dump_dir, exist_ok=True)
                im = s["image"].copy()
                im.thumbnail((1024, 1024))
                im.save(dump + ".jpg", quality=90)
                n_dumped += 1
            try:
                row["grad"] = grad_pass(model, processor, base, hd_extra, STATE["target"], s["gt"], off_params,
                                        dump=dump)
            except Exception as e:  # keep the run going; report
                print(f"[collapse] {gi} grad failed: {e}", flush=True)
                row["grad"] = None
        rows.append(row)
        print(f"[collapse] {gi} " + " ".join(f"{c}={int(row['ok'][c])}" for c in CFGS) +
              f" beta_or={row['beta']['oracle']:.2f} beta_real={row['beta']['real']:.2f}" +
              (f" grad_ratio={row['grad']['par_lm'] / max(row['grad']['par_box'], 1e-12):.3g}" if row.get("grad") else ""),
              flush=True)
        set_force_locs(model, None)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(args.out))), exist_ok=True)
        json.dump(rows, open(os.path.expanduser(args.out), "w"))
    summarize(rows)


if __name__ == "__main__":
    main()
