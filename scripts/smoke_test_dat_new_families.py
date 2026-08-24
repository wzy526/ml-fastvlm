#!/usr/bin/env python3
"""冒烟测试: Qwen3-VL / Qwen3.5 的 DAT 移植 (合成输入, 无数据依赖)。

验证项 (每个 family):
  1. convert_*_to_dat: 载入 base + 构建 DAT 层 (qwen3_5 校验 D 落在 full-attn 位)
  2. processor 双分辨率: min_pixels/max_pixels kwargs 在新 processor 上可用
  3. forward(labels) → loss 有限; backward → DAT 模块有梯度, 冻结的 base 无梯度
  4. no-HD 一致性: 不传 HD 输入时 DAT 模型 logits ≈ base 模型 (fallback 路径无损)
  5. generate: prefill 带 HD + decode 走 cache 路径不崩

用法 (pod):
  python scripts/smoke_test_dat_new_families.py --family qwen3_vl \
      --base-model /workspace/model_cache/Qwen3-VL-2B-Instruct
  python scripts/smoke_test_dat_new_families.py --family qwen3_5 \
      --base-model /workspace/model_cache/Qwen3.5-2B
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import argparse

import numpy as np
import torch
from PIL import Image


def make_test_image(w=1920, h=1080, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    # 加一些结构 (格子), 避免纯噪声
    arr[::64, :, :] = 255
    arr[:, ::64, :] = 0
    return Image.fromarray(arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=["qwen3_vl", "qwen3_5"])
    p.add_argument("--base-model", required=True)
    p.add_argument("--lr-pixels", type=int, default=256 * 32 * 32)
    p.add_argument("--hr-scale", type=int, default=3)
    args = p.parse_args()

    from transformers import AutoProcessor, AutoConfig

    if args.family == "qwen3_vl":
        from llava.model.language_model.modeling_qwen3_vl_dat import (
            convert_qwen3_vl_to_dat as convert_fn,
        )
        from transformers import Qwen3VLForConditionalGeneration as BaseCls
        base_cfg = AutoConfig.from_pretrained(args.base_model)
        n_layers = base_cfg.text_config.num_hidden_layers
        layers = ''.join('D' if i % 6 == 0 else 'L' for i in range(n_layers))
    else:
        from llava.model.language_model.modeling_qwen3_5_dat import (
            convert_qwen3_5_to_dat as convert_fn,
        )
        from transformers import Qwen3_5ForConditionalGeneration as BaseCls
        layers = 'auto'

    dat_extra_args = {
        'grid_size': 12,
        'off_ksize': 3,
        'off_grps': 8,
        'inter_size': 128,
        'hr_scale': args.hr_scale,
        'hd_proj': True,
        'layers': layers,
        'use_intention_branch': True,
        'intention_as_gate': True,
        'use_spatial_attn_guide': True,
        'hd_gate_init': None,
        'hd_gate_freeze': False,
        'inject_lr_image': False,
        'image_hd_for_question': False,
        'use_fused_vit': False,
        'use_shared_vit': False,
    }

    device = "cuda"
    dtype = torch.bfloat16

    # ---- 1. convert ----
    print(f"[1/5] convert {args.base_model} -> DAT ({args.family})")
    model = convert_fn(args.base_model, dat_extra_args, torch_dtype=dtype)
    model = model.to(device).eval()
    resolved_layers = model.config.dat_extra_args['layers']
    print(f"      layers={resolved_layers} (D count={resolved_layers.count('D')})")

    # ---- 2. processor 双分辨率 ----
    print("[2/5] processor dual-resolution")
    processor = AutoProcessor.from_pretrained(args.base_model)
    img = make_test_image()
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the fine details of this image."},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "It is a grid pattern over noise."},
        ]},
    ]
    tmpl_kwargs = {}
    if args.family == "qwen3_5":
        tmpl_kwargs["enable_thinking"] = False
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **tmpl_kwargs
    )
    lr_pix = args.lr_pixels
    hd_pix = lr_pix * args.hr_scale ** 2
    inputs = processor(images=[img], text=[text], return_tensors="pt",
                       padding=False, min_pixels=lr_pix, max_pixels=lr_pix)
    inputs_hd = processor(images=[img], text=["x"], return_tensors="pt",
                          padding=False, min_pixels=hd_pix, max_pixels=hd_pix)
    print(f"      LR thw={inputs['image_grid_thw'].tolist()}, "
          f"HD thw={inputs_hd['image_grid_thw'].tolist()}, "
          f"seq={inputs['input_ids'].shape[1]}")

    input_ids = inputs["input_ids"].to(device)
    labels = input_ids.clone()
    labels[:, :-8] = -100  # 最后 8 token 当 answer 段

    fwd_kwargs = dict(
        input_ids=input_ids,
        labels=labels,
        pixel_values=inputs["pixel_values"].to(device, dtype),
        image_grid_thw=inputs["image_grid_thw"].to(device),
        pixel_values_hd=inputs_hd["pixel_values"].to(device, dtype),
        image_grid_thw_hd=inputs_hd["image_grid_thw"].to(device),
    )
    if "mm_token_type_ids" in inputs:
        fwd_kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"].to(device)

    # ---- 3. forward + backward ----
    print("[3/5] forward(labels) + backward")
    from llava.model.language_model import modeling_qwen3_vl_dat as _m3
    freeze_mod = _m3
    if args.family == "qwen3_5":
        from llava.model.language_model import modeling_qwen3_5_dat as _m35
        freeze_mod = _m35
    freeze_mod.freeze_base_unfreeze_dat(model)
    model.train()
    out = model(**fwd_kwargs)
    loss = out.loss
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    print(f"      loss={loss.item():.4f}")
    loss.backward()
    n_dat_grad, n_base_grad = 0, 0
    for name, prm in model.named_parameters():
        if prm.grad is not None and prm.grad.abs().sum() > 0:
            if any(k in name for k in freeze_mod.DAT_KEYS_MATCH):
                n_dat_grad += 1
            else:
                n_base_grad += 1
    print(f"      DAT params with grad: {n_dat_grad}, base params with grad: {n_base_grad}")
    assert n_dat_grad > 0, "no DAT gradients!"
    assert n_base_grad == 0, "frozen base received gradients!"
    model.zero_grad(set_to_none=True)
    model.eval()

    # ---- 4. no-HD 一致性 ----
    print("[4/5] no-HD fallback vs base logits")
    with torch.no_grad():
        out_dat = model(
            input_ids=input_ids,
            pixel_values=fwd_kwargs["pixel_values"],
            image_grid_thw=fwd_kwargs["image_grid_thw"],
            **({"mm_token_type_ids": fwd_kwargs["mm_token_type_ids"]}
               if "mm_token_type_ids" in fwd_kwargs else {}),
        )
        base = BaseCls.from_pretrained(args.base_model, torch_dtype=dtype).to(device).eval()
        out_base = base(
            input_ids=input_ids,
            pixel_values=fwd_kwargs["pixel_values"],
            image_grid_thw=fwd_kwargs["image_grid_thw"],
            **({"mm_token_type_ids": fwd_kwargs["mm_token_type_ids"]}
               if "mm_token_type_ids" in fwd_kwargs else {}),
        )
        # 只比最后 32 个位置的 logits (足够灵敏, 避免全量显存)
        l_dat = out_dat.logits[:, -32:, :].float()
        l_base = out_base.logits[:, -32:, :].float()
        max_diff = (l_dat - l_base).abs().max().item()
        print(f"      max |logits_dat - logits_base| = {max_diff:.6f}")
        assert max_diff < 0.5, (
            f"no-HD fallback diverges from base (max_diff={max_diff}); "
            "attention fallback path is broken"
        )
        del base
        torch.cuda.empty_cache()

    # ---- 5. generate (prefill HD + decode cache) ----
    print("[5/5] generate with HD prefill")
    gen_text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What pattern do you see?"},
        ]}],
        tokenize=False, add_generation_prompt=True, **tmpl_kwargs
    )
    gen_inputs = processor(images=[img], text=[gen_text], return_tensors="pt",
                           padding=False, min_pixels=lr_pix, max_pixels=lr_pix)
    gen_kwargs = dict(
        input_ids=gen_inputs["input_ids"].to(device),
        pixel_values=gen_inputs["pixel_values"].to(device, dtype),
        image_grid_thw=gen_inputs["image_grid_thw"].to(device),
        pixel_values_hd=inputs_hd["pixel_values"].to(device, dtype),
        image_grid_thw_hd=inputs_hd["image_grid_thw"].to(device),
        max_new_tokens=16,
        do_sample=False,
    )
    if "mm_token_type_ids" in gen_inputs:
        gen_kwargs["mm_token_type_ids"] = gen_inputs["mm_token_type_ids"].to(device)
    with torch.no_grad():
        gen_out = model.generate(**gen_kwargs)
    new_tokens = gen_out[0, gen_inputs["input_ids"].shape[1]:]
    print(f"      generated: {processor.tokenizer.decode(new_tokens, skip_special_tokens=True)!r}")

    print(f"\n[PASS] {args.family} DAT smoke test OK")


if __name__ == "__main__":
    main()
