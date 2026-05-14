"""Verify hd_gate / DAT-init fix end-to-end via ``from_pretrained``.

Reproduces the model-construction path used by
``scripts/qwen2_5vl_adl_0514/exp9_full_1d5l_sa1b_ivcap.sh`` (i.e. the same
``--dat_*`` flags), then prints the actual loaded ``hd_gate``,
``conv_lr_dw``, ``conv_lr_proj``, ``conv_off_proj`` and ``proj_intention``
values on each DAT layer.

Pass criteria:
  * every ``layer.self_attn.hd_gate`` equals the requested ``hd_gate_init``
    (``-8.0`` for the 0514 run), to within fp32 precision;
  * conv weights have non-trivial std (Kaiming-normal, not zero / NaN);
  * ``proj_intention`` weight has non-trivial std (Xavier-uniform);
  * ``k_proj_hd`` / ``v_proj_hd`` are populated (post ``init_hd_proj_from_kv``)
    and roughly match the pretrained ``k_proj`` / ``v_proj`` they were
    copied from.

Run:
    cd /root/autodl-tmp/ml-fastvlm
    python scripts/qwen2_5vl_adl_0514/_verify_hd_gate_init.py
"""

from __future__ import annotations

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from llava.model.language_model.modeling_qwen2_5vl_dat import (  # noqa: E402
    Qwen2_5_VLAttentionDAT,
    Qwen2_5_VLDATConfig,
    Qwen2_5_VLDATForConditionalGeneration,
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct",
)
# Mirrors exp9_full_1d5l_sa1b_ivcap.sh (0514 re-run).
HD_GATE_INIT = float(os.environ.get("HD_GATE_INIT", "-8.0"))
DAT_LAYERS = os.environ.get(
    "DAT_LAYERS",
    "DLLLLLDLLLLLDLLLLLDLLLLLDLLLLLDLLLLL",
)
# Schema mirrors train_qwen_dat.py:2538-2552 exactly. ``off_ksize`` is not
# set by exp9.sh and falls back to the training-script default of 3.
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


def _stats(t: torch.Tensor) -> str:
    t = t.detach().float()
    return (
        f"shape={tuple(t.shape)} "
        f"mean={t.mean().item():+.4e} "
        f"std={t.std().item():.4e} "
        f"min={t.min().item():+.4e} "
        f"max={t.max().item():+.4e}"
    )


def main() -> None:
    print(f"[verify] MODEL_PATH    = {MODEL_PATH}")
    print(f"[verify] DAT_LAYERS    = {DAT_LAYERS}")
    print(f"[verify] hd_gate_init  = {HD_GATE_INIT}")
    print()

    config = Qwen2_5_VLDATConfig.from_pretrained(MODEL_PATH)
    config.use_dat = True
    config.dat_extra_args = DAT_EXTRA_ARGS

    model = Qwen2_5_VLDATForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float32,
    )
    model.init_hd_proj_from_kv()

    layers = model.model.language_model.layers
    dat_layer_idx = [i for i, c in enumerate(DAT_LAYERS) if c == "D"]
    print(f"[verify] DAT layer indices = {dat_layer_idx}\n")

    all_ok = True
    for i in dat_layer_idx:
        attn = layers[i].self_attn
        assert isinstance(attn, Qwen2_5_VLAttentionDAT), (
            f"layer {i} self_attn is not DAT: got {type(attn).__name__}"
        )

        hd_gate_val = float(attn.hd_gate.detach())
        hd_gate_ok = abs(hd_gate_val - HD_GATE_INIT) < 1e-4
        all_ok &= hd_gate_ok

        print(f"--- layer {i:2d} self_attn ---")
        print(f"  hd_gate           = {hd_gate_val:+.6f}   "
              f"(expected {HD_GATE_INIT}; ok={hd_gate_ok})")
        print(f"  conv_lr_dw.weight {_stats(attn.conv_lr_dw.weight)}")
        print(f"  conv_lr_proj.w    {_stats(attn.conv_lr_proj.weight)}")
        print(f"  conv_off_proj.w   {_stats(attn.conv_off_proj.weight)}")
        if isinstance(attn.proj_intention, torch.nn.Linear):
            print(f"  proj_intention.w  {_stats(attn.proj_intention.weight)}")
        # K-space alignment check: k_proj_hd should mirror k_proj after
        # init_hd_proj_from_kv (zero residual).
        diff_k = (attn.k_proj_hd.weight - attn.k_proj.weight).abs().max().item()
        diff_v = (attn.v_proj_hd.weight - attn.v_proj.weight).abs().max().item()
        print(f"  |k_proj_hd - k_proj|_inf = {diff_k:.3e}")
        print(f"  |v_proj_hd - v_proj|_inf = {diff_v:.3e}")
        print()

    print("=" * 60)
    if all_ok:
        print("[verify] PASS: every DAT layer hd_gate matches hd_gate_init.")
    else:
        print("[verify] FAIL: at least one DAT layer hd_gate is wrong.")
        sys.exit(1)


if __name__ == "__main__":
    main()
