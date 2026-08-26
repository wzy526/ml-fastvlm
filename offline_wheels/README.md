# Offline wheels

Pure-python wheels for clusters whose pip index (yum.tbsite.net) lacks them.

## fla (flash-linear-attention) — GatedDeltaNet kernels for Qwen3.5

Without fla, transformers' GatedDeltaNet layers fall back to a very slow
torch path (~13 s/it on the 0826 qwen3.5 pretrain).

**Use 0.3.2 on the OSS training cluster** (torch 2.5/2.6-era triton).
fla >= 0.4 ships `fla/ops/cp/` whose triton.autotune keys ('BT'/'USE_EXP2')
are not kernel arguments — old triton autotuners reject that at import time
with `ValueError: 'BT' is not in list`, which then breaks the transformers
qwen3_5 import entirely. 0.3.2 predates the cp module and imports cleanly.
transformers only needs FusedRMSNormGated + (chunk|fused_recurrent)_gated_
delta_rule, all present in 0.3.2.

```bash
python -m pip uninstall -y fla-core flash-linear-attention   # drop 0.5.0 if present
python -m pip install --no-deps \
    offline_wheels/fla_core-0.3.2-py3-none-any.whl \
    offline_wheels/flash_linear_attention-0.3.2-py3-none-any.whl
python -c "import fla; print('fla', fla.__version__)"
python -c "from fla.modules import FusedRMSNormGated; from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; print('fla API ok')"
```

The 0.5.0 wheels are kept for newer environments (torch >= 2.7, e.g. the
dev pod) — same install commands with the 0.5.0 filenames.

`--no-deps` is intentional: the real deps (torch, einops, transformers) are
already in the env; fla's own pins would otherwise make pip touch torch.
fla is triton-only — no CUDA compilation.

Note: `causal-conv1d` (optional extra speed for the GDN short conv) is NOT
shipped here — it needs a CUDA build matching the cluster's torch; skip it
unless profiling shows the conv is a bottleneck.
