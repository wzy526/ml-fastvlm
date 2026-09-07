#!/usr/bin/env python3
"""Weight-level HD-pathway collapse check for Qwen3.5-DAT checkpoints (CPU, seconds).

DAT layers init as a zero-init adapter: k_proj_hd = Kaiming, v_proj_hd = 0,
hd_input_layernorm = 1, conv_off_proj = 0. If training never pulled the HD
branch away from that init, the branch is content-blind by construction:

  v_proj_hd ~ 0  -> HD values are ~0 -> HD only DILUTES the LR output by
                    (1 - w_hd), identically for any HD content
  and with V = 0 the gradient reaching k_proj_hd is exactly zero, so K stays
  random. Classic zero-init trap.

Reports, per DAT layer:
  ||W||_F of k_proj_hd / v_proj_hd vs. the base k_proj / v_proj,
  ratio of k_proj_hd to its expected Kaiming init norm (sqrt(out_features)),
  hd_input_layernorm weight mean/std (init 1.0),
  conv_off_proj norm (init 0; offsets), effective rank of v_proj_hd.

Usage:
  python scripts/check_hd_weights_qwen35.py --model_path /path/to/merged_ckpt
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import torch

PAT = re.compile(
    r"layers\.(\d+)\.self_attn\.(k_proj_hd|v_proj_hd|hd_input_layernorm|"
    r"conv_off_proj|conv_lr_proj|conv_lr_dw|proj_intention|k_proj|v_proj|hd_gate)"
    r"(?:\.(weight|bias))?$"
)


def iter_tensors(model_path):
    """Yield (layer, module, kind, tensor) for DAT-relevant params.

    Accepts a merged/full ckpt dir (*.safetensors) OR a LoRA trainer dir
    (non_lora_trainables.bin), so trainer output can be diffed against the
    merged model to rule out a merge/save bug.
    """
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    nlt = os.path.join(model_path, "non_lora_trainables.bin")
    if files:
        from safetensors import safe_open
        for f in files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    m = PAT.search(k)
                    if m:
                        yield int(m.group(1)), m.group(2), (m.group(3) or "param"), sf.get_tensor(k)
    elif os.path.exists(nlt):
        sd = torch.load(nlt, map_location="cpu")
        for k, t in sd.items():
            m = PAT.search(k)
            if m:
                yield int(m.group(1)), m.group(2), (m.group(3) or "param"), t
    else:
        raise SystemExit(f"no *.safetensors or non_lora_trainables.bin under {model_path}")


def compare(layers_a, layers_b, keys=("k_proj_hd.weight", "v_proj_hd.weight",
                                       "conv_off_proj.weight", "hd_input_layernorm.weight")):
    print("\n==== A vs B tensor diff (rel = ||A-B|| / ||A||) ====")
    hdr = f"{'layer':>5} | " + " | ".join(f"{k.split('.')[0]:>18}" for k in keys)
    print(hdr); print("-" * len(hdr))
    for lid in sorted(set(layers_a) & set(layers_b)):
        if "k_proj_hd.weight" not in layers_a[lid]:
            continue  # not a DAT layer
        cells = []
        for k in keys:
            a, b = layers_a[lid].get(k), layers_b[lid].get(k)
            if a is None or b is None:
                cells.append(f"{'n/a':>18}"); continue
            a, b = a.float(), b.float()
            if a.shape != b.shape:
                cells.append(f"{'shape!':>18}"); continue
            d = (a - b).norm().item()
            rel = d / max(a.norm().item(), 1e-12)
            tag = "IDENTICAL" if d == 0.0 else f"rel {rel:.2e}"
            cells.append(f"{tag:>18}")
        print(f"{lid:>5} | " + " | ".join(cells))
    print("  IDENTICAL for k_proj_hd across training steps => K received ZERO updates")
    print("  (zero-init-V trap or frozen); a merge/save bug shows up as IDENTICAL between")
    print("  trainer ckpt and merged model only if the trainer ckpt is ALSO at init.")


def eff_rank(w):
    """Entropy-based effective rank of a 2-D weight (0 if all-zero)."""
    if w.numel() == 0 or float(w.abs().max()) == 0.0:
        return 0.0
    s = torch.linalg.svdvals(w.float())
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--compare", default=None,
                    help="second ckpt dir (merged model, full ckpt, or LoRA trainer dir "
                         "with non_lora_trainables.bin) to diff DAT tensors against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    layers = defaultdict(dict)
    for lid, name, kind, t in iter_tensors(args.model_path):
        layers[lid][f"{name}.{kind}"] = t
    layers_b = None
    if args.compare:
        layers_b = defaultdict(dict)
        for lid, name, kind, t in iter_tensors(args.compare):
            layers_b[lid][f"{name}.{kind}"] = t

    dat_layers = sorted(l for l in layers if "k_proj_hd.weight" in layers[l])
    if not dat_layers:
        raise SystemExit("no k_proj_hd found — is this a DAT ckpt with dat_hd_proj=True?")

    print(f"ckpt: {args.model_path}")
    print(f"DAT layers: {dat_layers}\n")
    hdr = (f"{'layer':>5} | {'|Wk_hd|':>8} {'|Wk|':>8} {'k_hd/init':>9} | "
           f"{'|Wv_hd|':>8} {'|Wv|':>8} {'v_hd/v':>7} {'erank_v':>7} | "
           f"{'ln_w mean':>9} {'ln_w std':>8} | {'|off|':>8} {'gate':>6}")
    print(hdr); print("-" * len(hdr))

    report = {}
    for lid in dat_layers:
        L = layers[lid]
        wk_hd = L["k_proj_hd.weight"].float()
        wv_hd = L["v_proj_hd.weight"].float()
        wk = L.get("k_proj.weight", torch.zeros(1)).float()
        wv = L.get("v_proj.weight", torch.zeros(1)).float()
        nk_hd, nv_hd = wk_hd.norm().item(), wv_hd.norm().item()
        nk, nv = wk.norm().item(), wv.norm().item()
        # kaiming_normal_(nonlinearity='linear'): std = 1/sqrt(fan_in)
        # -> E||W||_F^2 = out * in * (1/in) = out_features
        k_init_expect = math.sqrt(wk_hd.shape[0])
        ln = L.get("hd_input_layernorm.weight")
        ln_mean = ln.float().mean().item() if ln is not None else float("nan")
        ln_std = ln.float().std().item() if ln is not None else float("nan")
        off = L.get("conv_off_proj.weight")
        n_off = off.float().norm().item() if off is not None else float("nan")
        gate = L.get("hd_gate.param")
        gate_s = f"{torch.sigmoid(gate.float()).item():.3f}" if gate is not None else "  none"
        er_v = eff_rank(wv_hd)

        report[lid] = dict(k_hd_norm=nk_hd, k_norm=nk, k_hd_over_init=nk_hd / k_init_expect,
                           v_hd_norm=nv_hd, v_norm=nv,
                           v_hd_over_v=(nv_hd / nv if nv > 0 else float("nan")),
                           v_hd_eff_rank=er_v, ln_mean=ln_mean, ln_std=ln_std,
                           off_norm=n_off, gate=gate_s.strip())
        print(f"{lid:>5} | {nk_hd:>8.2f} {nk:>8.2f} {nk_hd / k_init_expect:>9.3f} | "
              f"{nv_hd:>8.3f} {nv:>8.2f} {report[lid]['v_hd_over_v']:>7.3f} {er_v:>7.1f} | "
              f"{ln_mean:>9.3f} {ln_std:>8.3f} | {n_off:>8.4f} {gate_s:>6}")

    print("\nhow to read:")
    print("  v_hd/v  ~0 (e.g. <0.05)  -> v_proj_hd never left zero-init: HD values ~0, branch")
    print("                              is pure dilution, content-blind by construction")
    print("  k_hd/init ~1.0            -> k_proj_hd still at random Kaiming init (no K learning;")
    print("                              expected when V~0 since dK ∝ V)")
    print("  erank_v small (<10)       -> whatever V learned is a near-constant direction")
    print("  |off| ~0                  -> offsets never learned: sampling = fixed regular grid")

    if layers_b is not None:
        compare(layers, layers_b)

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
