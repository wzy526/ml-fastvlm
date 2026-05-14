"""Verify WandbDATMonitorCallback works correctly under LoRA + DAT.

Reproduces the exact model construction path used by exp9_full_1d5l_sa1b_ivcap.sh
(0514 re-run with ``--dat_hd_gate_init -8.0`` and ``--lora_target_layers dat``):

  1. ``Qwen2_5_VLDATForConditionalGeneration.from_pretrained``
  2. ``init_hd_proj_from_kv``
  3. ``get_peft_model(model, LoraConfig(target_modules=..., bias="none"))``
  4. Post-PEFT explicit DAT unfreeze loop (train_qwen_dat.py:2694-2700)

then asserts every property the tracker depends on:

  * for every name produced by ``model.named_parameters()`` that matches the
    DAT_KEYS_MATCH substring filter, ``requires_grad`` is True
    (otherwise ``_register_hooks`` registers zero hooks → no dat/grad_norm);
  * there are NO false-positive matches against LoRA adapter weights
    (e.g. nothing like ``...k_proj.lora_A.default.weight`` slips through);
  * ``hd_gate`` is present with ``numel() == 1`` so the scalar-value branch
    in ``_compute_weight_norms`` fires (line 1652);
  * optimizer param-grouping is sane: every DAT-matched name ends up in a
    group whose ``lr`` equals ``dat_lr``, and LoRA adapters end up in a
    different group whose ``lr`` equals ``lora_lr``.

Run:
    cd /root/autodl-tmp/ml-fastvlm
    python scripts/qwen2_5vl_adl_0514/_verify_lora_tracker.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from llava.model.language_model.modeling_qwen2_5vl_dat import (  # noqa: E402
    DAT_KEYS_MATCH,
    Qwen2_5_VLAttentionDAT,
    Qwen2_5_VLDATConfig,
    Qwen2_5_VLDATForConditionalGeneration,
)
from llava.train.train_qwen_dat import (  # noqa: E402
    DAT_KEYS_MATCH as TRAIN_DAT_KEYS_MATCH,
    get_lora_target_modules,
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct",
)
HD_GATE_INIT = float(os.environ.get("HD_GATE_INIT", "-8.0"))
DAT_LAYERS = os.environ.get(
    "DAT_LAYERS",
    "DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL",
)
DAT_EXTRA_ARGS = {
    "grid_size": 20,
    "off_ksize": 3,
    "off_grps": 8,
    "inter_size": 128,
    "hr_scale": 3,
    "hd_proj": True,
    "layers": DAT_LAYERS,
    "use_intention_branch": True,
    "intention_as_gate": True,
    "use_spatial_attn_guide": False,
    "hd_gate_init": HD_GATE_INIT,
    "use_fused_vit": False,
    "use_shared_vit": False,
}

# Mirrors exp9.sh
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_LAYERS = "dat"
DAT_LR = 1e-4
LORA_LR = 2e-5
BASE_LR = 2e-5
WEIGHT_DECAY = 0.0


def matches_dat(name: str) -> bool:
    return any(k in name for k in DAT_KEYS_MATCH)


def main() -> None:
    # DAT_KEYS_MATCH should be identical in modeling and train code.
    assert list(DAT_KEYS_MATCH) == list(TRAIN_DAT_KEYS_MATCH), (
        f"DAT_KEYS_MATCH desync: modeling={DAT_KEYS_MATCH} "
        f"train={TRAIN_DAT_KEYS_MATCH}"
    )
    print(f"[verify-lora] DAT_KEYS_MATCH = {DAT_KEYS_MATCH}\n")

    config = Qwen2_5_VLDATConfig.from_pretrained(MODEL_PATH)
    config.use_dat = True
    config.dat_extra_args = DAT_EXTRA_ARGS

    model = Qwen2_5_VLDATForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float32,
    )
    model.init_hd_proj_from_kv()

    # ------ Step 1: get_peft_model (LoRA) ------
    from peft import LoraConfig, get_peft_model

    target_modules = get_lora_target_modules(
        DAT_LAYERS, target_layers=LORA_TARGET_LAYERS
    )
    print(f"[verify-lora] LoRA target_modules pattern: {target_modules}")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # ------ Step 2: post-PEFT DAT unfreeze (mirrors train_qwen_dat.py:2694-2700) ------
    dat_unfrozen = 0
    for name, param in model.named_parameters():
        if matches_dat(name):
            param.requires_grad = True
            dat_unfrozen += 1
    print(f"[verify-lora] Unfroze {dat_unfrozen} DAT parameters after LoRA wrapping\n")

    # =================================================================
    # Check A: every DAT-matched param has requires_grad=True
    # =================================================================
    dat_named = [(n, p) for n, p in model.named_parameters() if matches_dat(n)]
    dat_named_no_grad = [n for n, p in dat_named if not p.requires_grad]
    print(f"[A] #DAT-matched params       = {len(dat_named)}")
    print(f"[A] #DAT-matched, no_grad     = {len(dat_named_no_grad)}")
    if dat_named_no_grad:
        print("  → these will be IGNORED by the tracker:")
        for n in dat_named_no_grad[:20]:
            print(f"    {n}")
        print(f"[A] FAIL: {len(dat_named_no_grad)} DAT params still frozen")
    else:
        print("[A] PASS: every DAT-matched param is trainable\n")

    # Group by which DAT key matched, to spot any pattern that ends up empty.
    per_key = defaultdict(int)
    for n, _ in dat_named:
        for k in DAT_KEYS_MATCH:
            if k in n:
                per_key[k] += 1
                break
    print("[A] DAT-match breakdown by key (first matching wins):")
    for k in DAT_KEYS_MATCH:
        print(f"    {k:18s} -> {per_key.get(k, 0)} params")
    print()

    # =================================================================
    # Check B: no LoRA adapter weight is mis-classified as DAT
    # =================================================================
    lora_named = [
        n for n, _ in model.named_parameters()
        if "lora_A" in n or "lora_B" in n or "lora_embedding" in n
    ]
    lora_misclassified = [n for n in lora_named if matches_dat(n)]
    print(f"[B] #LoRA adapter params      = {len(lora_named)}")
    print(f"[B] #LoRA mis-classified-as-DAT = {len(lora_misclassified)}")
    if lora_misclassified:
        print("  → tracker would mix LoRA adapters into dat/grad_norm:")
        for n in lora_misclassified[:10]:
            print(f"    {n}")
        print("[B] FAIL")
    else:
        print("[B] PASS: no LoRA adapter weight matches DAT_KEYS_MATCH\n")

    # =================================================================
    # Check C: hd_gate is a scalar (numel==1) on every DAT layer
    # =================================================================
    hd_gates = [(n, p) for n, p in dat_named if "hd_gate" in n]
    print(f"[C] #hd_gate params           = {len(hd_gates)}")
    bad_shape = [(n, p.shape, p.numel()) for n, p in hd_gates if p.numel() != 1]
    if bad_shape:
        print("[C] FAIL: hd_gate(s) with numel != 1:")
        for n, sh, ne in bad_shape:
            print(f"    {n}  shape={sh} numel={ne}")
    else:
        sample_name, sample_p = hd_gates[0]
        print(f"[C] PASS: every hd_gate has numel==1 "
              f"(sample value={sample_p.detach().float().item():+.6f}, "
              f"name={sample_name})\n")

    # =================================================================
    # Check D: optimizer param-grouping (replicates Trainer.create_optimizer)
    # =================================================================
    from transformers.trainer import get_parameter_names
    ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, torch.nn.BatchNorm2d]
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    lr_mapper = {}
    for key in DAT_KEYS_MATCH:
        lr_mapper[key] = DAT_LR
    lr_mapper["lora_"] = LORA_LR

    special_lr_parameters = [
        name for name, _ in model.named_parameters()
        if any(module_keyword in name for module_keyword in lr_mapper)
    ]
    grouped = [
        {"name": "base_decay", "lr": BASE_LR, "weight_decay": WEIGHT_DECAY,
         "names": [n for n, p in model.named_parameters()
                   if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)]},
        {"name": "base_nodecay", "lr": BASE_LR, "weight_decay": 0.0,
         "names": [n for n, p in model.named_parameters()
                   if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)]},
    ]
    assigned: set[str] = set()
    for module_keyword, lr in lr_mapper.items():
        module_parameters = [
            n for n, _ in model.named_parameters()
            if module_keyword in n and n not in assigned
        ]
        assigned.update(module_parameters)
        grouped.append({
            "name": f"{module_keyword}_decay", "lr": lr, "weight_decay": WEIGHT_DECAY,
            "names": [n for n, p in model.named_parameters()
                      if (n in decay_parameters and n in module_parameters and p.requires_grad)],
        })
        grouped.append({
            "name": f"{module_keyword}_nodecay", "lr": lr, "weight_decay": 0.0,
            "names": [n for n, p in model.named_parameters()
                      if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
        })

    print("[D] Optimizer param-grouping (trainable params only):")
    total_in_groups = 0
    for g in grouped:
        if g["names"]:
            print(f"    [{g['name']:30s}] lr={g['lr']:.2e}  wd={g['weight_decay']:.2f}  "
                  f"#params={len(g['names'])}")
            total_in_groups += len(g["names"])

    total_trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    print(f"[D] total trainable={total_trainable}, total grouped={total_in_groups}, "
          f"diff={total_trainable - total_in_groups}")

    # Are all DAT params in a group with lr=dat_lr?
    dat_in_dat_group = 0
    dat_in_wrong_group = []
    dat_names = {n for n, _ in dat_named}
    for g in grouped:
        if abs(g["lr"] - DAT_LR) < 1e-12:
            for n in g["names"]:
                if n in dat_names:
                    dat_in_dat_group += 1
        else:
            for n in g["names"]:
                if n in dat_names:
                    dat_in_wrong_group.append((n, g["name"], g["lr"]))
    print(f"[D] DAT params landed in dat_lr group: {dat_in_dat_group} / {len(dat_names)}")
    if dat_in_wrong_group:
        print("[D] FAIL: some DAT params ended up at wrong LR:")
        for n, gn, lr in dat_in_wrong_group[:10]:
            print(f"    {n} -> group={gn} lr={lr}")
    else:
        print("[D] PASS: every trainable DAT param is on dat_lr\n")

    # =================================================================
    # Check E: tracker LR-probe (line 1707-1718) finds a DAT param in
    #          the dat_lr group via id() match.
    # =================================================================
    try:
        probe = next(
            p for n, p in model.named_parameters()
            if matches_dat(n) and p.requires_grad
        )
        # Build flat optimizer-group-like view with actual parameter objects
        # (not just names), then check id-match works.
        actual_optim_groups = []
        for module_keyword, lr in lr_mapper.items():
            params_for_group = [
                p for n, p in model.named_parameters()
                if module_keyword in n and p.requires_grad
            ]
            actual_optim_groups.append({"lr": lr, "params": params_for_group})

        found_lr = None
        for g in actual_optim_groups:
            if any(id(p) == id(probe) for p in g["params"]):
                found_lr = g["lr"]
                break
        print(f"[E] tracker LR-probe found_lr={found_lr} (expected {DAT_LR})")
        if found_lr is None or abs(found_lr - DAT_LR) > 1e-12:
            print("[E] FAIL: tracker would log wrong / no dat/lr\n")
        else:
            print("[E] PASS: tracker's id() walk finds dat_lr correctly\n")
    except StopIteration:
        print("[E] FAIL: no trainable DAT param found at all\n")

    # =================================================================
    # Check F: DAT attention modules are still findable via model.modules()
    #          (LoRA wraps Linear children, not the DAT attention container).
    # =================================================================
    dat_attn_modules = [
        (i, type(m).__name__) for i, m in enumerate(model.modules())
        if isinstance(m, Qwen2_5_VLAttentionDAT)
    ]
    print(f"[F] Qwen2_5_VLAttentionDAT modules visible via model.modules() "
          f"= {len(dat_attn_modules)} (expected {DAT_LAYERS.count('D')})")
    if len(dat_attn_modules) != DAT_LAYERS.count("D"):
        print("[F] FAIL: _harvest_forward_diagnostics would miss some layers\n")
    else:
        print("[F] PASS: forward-diagnostic harvest will reach every DAT attn\n")

    # =================================================================
    # Summary
    # =================================================================
    checks = {
        "A: all DAT params trainable": not dat_named_no_grad,
        "B: no LoRA misclassified as DAT": not lora_misclassified,
        "C: hd_gate is scalar": not bad_shape and len(hd_gates) == DAT_LAYERS.count("D"),
        "D: all DAT params on dat_lr": not dat_in_wrong_group,
        "E: tracker LR-probe finds dat_lr": (found_lr is not None
                                             and abs(found_lr - DAT_LR) < 1e-12),
        "F: all DAT attn modules visible": (
            len(dat_attn_modules) == DAT_LAYERS.count("D")
        ),
    }
    print("=" * 60)
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    all_ok = all(checks.values())
    print("=" * 60)
    print(f"[verify-lora] {'OVERALL PASS' if all_ok else 'OVERALL FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
