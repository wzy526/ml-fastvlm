#!/usr/bin/env python3
"""把 base Qwen2.5-VL 转成 DAT 版并存盘 (DAT 模块随机初始化, 测速用)。

用法 (pod):
  python scripts/make_dat_ckpt.py \
      --base-model /workspace/model_cache/Qwen2.5-VL-3B-Instruct \
      --output /workspace/model_cache/Qwen2.5-VL-3B-Instruct-DAT-rand
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse

import torch

DAT_EXTRA_ARGS = {
    'grid_size': 20,
    'off_ksize': 3,
    'off_grps': 8,
    'inter_size': 128,
    'hr_scale': 3,
    'hd_proj': True,
    'use_intention_branch': True,
    'intention_as_gate': True,
    'use_spatial_attn_guide': False,
    'hd_gate_init': None,
    'hd_gate_freeze': False,
    'inject_lr_image': False,
    'image_hd_for_question': False,
    'use_fused_vit': False,
    'use_shared_vit': False,
}


def main():
    p = argparse.ArgumentParser()
    _cache = os.environ.get("MODEL_CACHE", "/workspace/model_cache")
    p.add_argument("--base-model", default=os.path.join(_cache, "Qwen2.5-VL-3B-Instruct"))
    p.add_argument("--output", default=os.path.join(_cache, "Qwen2.5-VL-3B-Instruct-DAT-rand"))
    args = p.parse_args()

    from transformers import AutoConfig, AutoProcessor
    from llava.model.language_model.modeling_qwen2_5vl_dat import convert_qwen2_5vl_to_dat

    base_cfg = AutoConfig.from_pretrained(args.base_model)
    n_layers = getattr(base_cfg, 'num_hidden_layers', None) or \
               base_cfg.text_config.num_hidden_layers
    dat_extra_args = dict(DAT_EXTRA_ARGS)
    dat_extra_args['layers'] = ''.join(
        'D' if i % 6 == 0 else 'L' for i in range(n_layers))

    print(f"[convert] {args.base_model} ({n_layers} layers, "
          f"pattern={dat_extra_args['layers']})")
    model = convert_qwen2_5vl_to_dat(args.base_model, dat_extra_args,
                                     torch_dtype=torch.bfloat16)

    n_total = sum(x.numel() for x in model.parameters()) / 1e9
    print(f"[save] params={n_total:.2f}B -> {args.output}")
    model.save_pretrained(args.output)
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.output)

    # 回读验证 save/load 对称 (conversion mapping 已注册)
    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        Qwen2_5_VLDATForConditionalGeneration,
    )
    reloaded = Qwen2_5_VLDATForConditionalGeneration.from_pretrained(
        args.output, torch_dtype=torch.bfloat16)
    n_reload = sum(x.numel() for x in reloaded.parameters()) / 1e9
    assert abs(n_reload - n_total) < 1e-6, f"参数量不一致: {n_reload} vs {n_total}"
    print(f"[verify] reload OK, params={n_reload:.2f}B")


if __name__ == "__main__":
    main()
