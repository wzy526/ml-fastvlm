#!/usr/bin/env python3
"""Diagnostic: does ``use_spatial_attn_guide`` actually do anything at inference?

Load a merged DAT ckpt, print:
  1. config.dat_extra_args after load (verifies edited config.json was picked up)
  2. per-layer self.use_spatial_attn_guide for every DAT layer
  3. logits diff when the flag is toggled at runtime on the loaded weights
  4. spatial_attn_guide tensor stats (mean/std/min/max) -- if mean ~ 1.0 with
     small std, the multiplication is a near-identity and the flag is a no-op
     for this checkpoint regardless of what config.json says.

Run:
    cd /root/autodl-tmp/ml-fastvlm
    /root/miniconda3/envs/vldat/bin/python scripts/dbg_spatial_attn_guide.py \
        --ckpt /root/autodl-tmp/vldat_experiments/0427_q25vl_sepvit_z3_lrfirst_baseline_no_kd-merged \
        --image /root/autodl-tmp/models_data/sft_data/vstar_bench/direct_attributes/sa_38195.jpg
"""
import argparse
import os
import sys
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--prompt', default="What is in this image?")
    ap.add_argument('--max_pixels', type=int, default=9031680)
    ap.add_argument('--min_pixels', type=int, default=28224)
    ap.add_argument('--lr_max_pixels', type=int, default=501760)
    ap.add_argument('--lr_min_pixels', type=int, default=200704)
    args = ap.parse_args()

    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        Qwen2_5_VLDATConfig,
        Qwen2_5_VLDATForConditionalGeneration,
        Qwen2_5_VLAttentionDAT,
    )
    from transformers import AutoProcessor

    print("=" * 70)
    print(f"Loading config from {args.ckpt}")
    cfg = Qwen2_5_VLDATConfig.from_pretrained(args.ckpt)
    print(f"  config.dat_extra_args = {cfg.dat_extra_args}")
    print(f"  config.dat_extra_args.use_spatial_attn_guide = "
          f"{cfg.dat_extra_args.get('use_spatial_attn_guide', '<missing>')}")

    print("=" * 70)
    print(f"Loading model from {args.ckpt} (bfloat16, eager attn for clean logits)")
    model = Qwen2_5_VLDATForConditionalGeneration.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval().cuda()

    # 2. Per-layer flag
    print("\n[layer] DAT attn layers and their use_spatial_attn_guide flag:")
    dat_layer_indices = []
    for i, layer in enumerate(model.model.language_model.layers):
        if isinstance(layer.self_attn, Qwen2_5_VLAttentionDAT):
            dat_layer_indices.append(i)
            print(f"  layer[{i:>2}].self_attn.use_spatial_attn_guide = "
                  f"{layer.self_attn.use_spatial_attn_guide}, "
                  f"use_intention_branch = {layer.self_attn.use_intention_branch}")
    print(f"  total DAT layers: {len(dat_layer_indices)}")

    # 3. Build a real input pair (LR + HR) using LR-first geometry from training.
    img = Image.open(args.image).convert("RGB")
    print(f"\n[input] image: {args.image}  size: {img.size}")

    proc = AutoProcessor.from_pretrained(
        args.ckpt, max_pixels=args.lr_max_pixels, min_pixels=args.lr_min_pixels,
    )
    hr_proc = AutoProcessor.from_pretrained(
        args.ckpt, max_pixels=100_000_000, min_pixels=784,
    )
    text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{args.prompt}<|im_end|>\n<|im_start|>assistant\n")

    lr_inputs = proc(images=[img], text=[text], return_tensors="pt", padding=False,
                     min_pixels=args.lr_min_pixels, max_pixels=args.lr_max_pixels)
    lr_thw = lr_inputs["image_grid_thw"][0]
    FACTOR = 28
    lr_h_px = int(lr_thw[1].item()) * FACTOR
    lr_w_px = int(lr_thw[2].item()) * FACTOR
    lr_pixels = lr_h_px * lr_w_px
    import math
    orig_pixels = img.width * img.height
    hd_target = lr_pixels * (cfg.dat_extra_args['hr_scale'] ** 2)
    hd_target = min(hd_target, orig_pixels, args.max_pixels)
    aspect = img.width / img.height
    hd_h = max(FACTOR, (int(math.sqrt(hd_target / aspect)) // FACTOR) * FACTOR)
    hd_w = max(FACTOR, (int(int(math.sqrt(hd_target / aspect)) * aspect) // FACTOR) * FACTOR)
    hd_total = hd_h * hd_w
    hr_inputs = hr_proc(images=[img], text=["<|im_start|>"], return_tensors="pt",
                        padding=False, min_pixels=hd_total, max_pixels=hd_total)
    print(f"  LR thw = {lr_inputs['image_grid_thw'].tolist()}, lr_pixels = {lr_pixels}")
    print(f"  HR thw = {hr_inputs['image_grid_thw'].tolist()}, hd_total = {hd_total}")

    fwd_kwargs = dict(
        input_ids=lr_inputs["input_ids"].cuda(),
        attention_mask=lr_inputs["attention_mask"].cuda(),
        pixel_values=lr_inputs["pixel_values"].to(torch.bfloat16).cuda(),
        image_grid_thw=lr_inputs["image_grid_thw"].cuda(),
        pixel_values_hd=hr_inputs["pixel_values"].to(torch.bfloat16).cuda(),
        image_grid_thw_hd=hr_inputs["image_grid_thw"].cuda(),
        use_cache=False,
    )

    # 4. Hook spatial_attn_guide tensor inside one DAT layer.
    captured = {}

    def hook_capture(module, inputs, output):
        # Wrap forward to capture intermediate. Cleanest way: monkey-patch forward
        # of one DAT attn layer to also stash the spatial_attn_guide.
        pass

    # Monkey-patch ONE DAT layer to capture spatial_attn_guide stats.
    target_layer_idx = dat_layer_indices[0]
    target_attn = model.model.language_model.layers[target_layer_idx].self_attn
    orig_forward = target_attn.forward
    import einops
    import torch.nn.functional as F

    def patched_forward(self, *fwd_args, **fwd_kw):
        # Capture early state by intercepting the relevant method.
        # Easier: just rerun the spatial_attn_guide manually by hooking via a
        # forward_pre_hook ... actually no, we need to be inside the forward.
        # Simpler trick: temporarily set self._capture = {} and copy-paste a
        # tiny version of the relevant lines. But that requires knowing
        # query_states etc. -- they're computed inside.
        # Instead: register an op hook on F.adaptive_avg_pool2d... too brittle.
        # Cleanest: just enable the guide branch manually and inspect via a
        # one-liner.
        return orig_forward(*fwd_args, **fwd_kw)

    # Actually a pragmatic approach: patch the GLOBAL F.adaptive_avg_pool2d
    # used in this module so we can spy on its outputs.
    import llava.model.language_model.modeling_qwen2_5vl_dat as M
    orig_pool = M.F.adaptive_avg_pool2d
    grid_size = cfg.dat_extra_args['grid_size']
    pool_calls = []

    def spy_pool(x, target_size):
        out = orig_pool(x, target_size)
        # The spatial_attn pooling is the only pool that goes to (grid_size, grid_size)
        # with input dtype float (we cast spatial_attn to .float()).
        if (isinstance(target_size, tuple) and target_size == (grid_size, grid_size)
                and x.dtype == torch.float32 and x.dim() == 4 and x.shape[1] == 1):
            pool_calls.append(out.detach().clone())
        return out

    # 4a. Run with use_spatial_attn_guide=True
    print("\n" + "=" * 70)
    print("[run-A] forcing use_spatial_attn_guide = True on every DAT layer")
    for i in dat_layer_indices:
        model.model.language_model.layers[i].self_attn.use_spatial_attn_guide = True
    M.F.adaptive_avg_pool2d = spy_pool
    pool_calls.clear()
    with torch.no_grad():
        out_A = model(**fwd_kwargs)
    M.F.adaptive_avg_pool2d = orig_pool
    logits_A = out_A.logits.detach().clone()
    print(f"  forward done; pool_calls captured = {len(pool_calls)}")
    if pool_calls:
        for k, t in enumerate(pool_calls[:6]):
            t = t.float()
            t_per_image = t * (lr_h_px // FACTOR) * (lr_w_px // FACTOR)  # × (lr_h*lr_w in merged-token units... actually wait
            # spatial_attn_guide is already × (lr_h * lr_w) where lr_h * lr_w
            # is in "merged token grid" -- but wait, in the code lr_h, lr_w are
            # *patches* (14-px), so lr_h * lr_w refers to 14-px-patch counts.
            # Actually re-read: see modeling_qwen2_5vl_dat.py line 642 -- lr_h, lr_w
            # are derived from image_grid_thw which is in MERGED tokens.
            # The * (lr_h * lr_w) makes mean ~= 1.0 by construction (softmax sums
            # to 1, * adaptive_pool * (lr_h * lr_w) keeps mean = 1).
            print(f"    pool[{k}] shape={tuple(t.shape)}  "
                  f"mean={t.mean().item():.4f}  std={t.std().item():.4f}  "
                  f"min={t.min().item():.4f}  max={t.max().item():.4f}")
        # Note: spatial_attn (pre-pool, post-softmax) sums to 1; pool result
        # mean = 1 / (grid_size^2). Multiplying by lr_h*lr_w (NOT shown in
        # spy capture) makes the FINAL guide tensor ~= 1.0 mean.

    # 4b. Run with use_spatial_attn_guide=False
    print("\n[run-B] forcing use_spatial_attn_guide = False on every DAT layer")
    for i in dat_layer_indices:
        model.model.language_model.layers[i].self_attn.use_spatial_attn_guide = False
    pool_calls.clear()
    with torch.no_grad():
        out_B = model(**fwd_kwargs)
    logits_B = out_B.logits.detach().clone()

    # Compare
    diff = (logits_A.float() - logits_B.float()).abs()
    print("\n" + "=" * 70)
    print("[compare] |logits_A - logits_B|")
    print(f"  max abs diff:  {diff.max().item():.6e}")
    print(f"  mean abs diff: {diff.mean().item():.6e}")
    print(f"  num exact-equal positions: "
          f"{(logits_A == logits_B).sum().item()} / {logits_A.numel()}")
    # Greedy argmax compare on the last-token logits (next-token predictions).
    argmax_A = logits_A[0, -1].argmax().item()
    argmax_B = logits_B[0, -1].argmax().item()
    print(f"  argmax_next_token A={argmax_A}  B={argmax_B}  "
          f"{'SAME' if argmax_A == argmax_B else 'DIFF'}")

    if diff.max().item() < 1e-7:
        print("\n>>> CONCLUSION: flag toggling is a NO-OP at runtime. <<<")
        print(">>> Either the branch is being short-circuited elsewhere, or")
        print(">>> spatial_guide_rep ≡ 1 (which would make the multiply identity).")
    else:
        print("\n>>> CONCLUSION: flag IS effective at runtime; logits change.")
        print(">>> Same vstar accuracy is then a benchmark-granularity collision.")


if __name__ == '__main__':
    main()
