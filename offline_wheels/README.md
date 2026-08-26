# Offline wheels

Pure-python wheels for clusters whose pip index (yum.tbsite.net) lacks them.

## fla (flash-linear-attention 0.5.0) — GatedDeltaNet kernels for Qwen3.5

Without fla, transformers' GatedDeltaNet layers fall back to a very slow
torch path (~13 s/it on the 0826 qwen3.5 pretrain). Install with:

```bash
pip install --no-deps \
    offline_wheels/fla_core-0.5.0-py3-none-any.whl \
    offline_wheels/flash_linear_attention-0.5.0-py3-none-any.whl
python -c "import fla; print('fla', fla.__version__)"
```

`--no-deps` is intentional: the real deps (torch, einops, transformers) are
already in the env, and fla-core's `torch>=2.7.0` pin would otherwise make
pip try to touch torch. fla itself is triton-only — no CUDA compilation.

Note: `causal-conv1d` (optional extra speed for the GDN short conv) is NOT
shipped here — it needs a CUDA build matching the cluster's torch; skip it
unless profiling shows the conv is a bottleneck.
