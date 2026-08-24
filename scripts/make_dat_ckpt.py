#!/usr/bin/env python3
"""把 base Qwen VLM 转成 DAT 版并存盘 (DAT 模块随机初始化, 测速用)。

支持 --family qwen2_5_vl / qwen3_vl / qwen3_5。

用法 (pod):
  python scripts/make_dat_ckpt.py \
      --base-model /workspace/model_cache/Qwen2.5-VL-3B-Instruct \
      --output /workspace/model_cache/Qwen2.5-VL-3B-Instruct-DAT-rand

  python scripts/make_dat_ckpt.py --family qwen3_5 \
      --base-model /workspace/model_cache/Qwen3.5-2B \
      --output /workspace/model_cache/Qwen3.5-2B-DAT-rand
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

_FAMILY_IMPORTS = {
    'qwen2_5_vl': ('llava.model.language_model.modeling_qwen2_5vl_dat',
                   'convert_qwen2_5vl_to_dat',
                   'Qwen2_5_VLDATForConditionalGeneration'),
    'qwen3_vl':   ('llava.model.language_model.modeling_qwen3_vl_dat',
                   'convert_qwen3_vl_to_dat',
                   'Qwen3VLDATForConditionalGeneration'),
    'qwen3_5':    ('llava.model.language_model.modeling_qwen3_5_dat',
                   'convert_qwen3_5_to_dat',
                   'Qwen3_5DATForConditionalGeneration'),
}


def main():
    p = argparse.ArgumentParser()
    _cache = os.environ.get("MODEL_CACHE", "/workspace/model_cache")
    p.add_argument("--family", default="qwen2_5_vl", choices=sorted(_FAMILY_IMPORTS))
    p.add_argument("--base-model", default=os.path.join(_cache, "Qwen2.5-VL-3B-Instruct"))
    p.add_argument("--output", default=os.path.join(_cache, "Qwen2.5-VL-3B-Instruct-DAT-rand"))
    p.add_argument("--layers", default=None,
                   help="Explicit D/L pattern. Default: every 6th layer for "
                        "qwen2_5_vl/qwen3_vl; 'auto' (all full-attn layers) for qwen3_5.")
    args = p.parse_args()

    import importlib
    from transformers import AutoConfig, AutoProcessor

    mod_name, convert_name, cls_name = _FAMILY_IMPORTS[args.family]
    mod = importlib.import_module(mod_name)
    convert_fn = getattr(mod, convert_name)
    model_cls = getattr(mod, cls_name)

    base_cfg = AutoConfig.from_pretrained(args.base_model)
    text_cfg = getattr(base_cfg, 'text_config', base_cfg)
    n_layers = text_cfg.num_hidden_layers

    dat_extra_args = dict(DAT_EXTRA_ARGS)
    if args.layers is not None:
        dat_extra_args['layers'] = args.layers
    elif args.family == 'qwen3_5':
        # Hybrid arch: anchor D layers to the full_attention positions.
        dat_extra_args['layers'] = 'auto'
    else:
        dat_extra_args['layers'] = ''.join(
            'D' if i % 6 == 0 else 'L' for i in range(n_layers))

    print(f"[convert] {args.base_model} ({args.family}, {n_layers} layers, "
          f"pattern={dat_extra_args['layers']})")
    model = convert_fn(args.base_model, dat_extra_args, torch_dtype=torch.bfloat16)

    n_total = sum(x.numel() for x in model.parameters()) / 1e9
    print(f"[save] params={n_total:.2f}B -> {args.output}")
    model.save_pretrained(args.output)
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.output)

    # 回读验证 save/load 对称 (conversion mapping 已注册)
    reloaded = model_cls.from_pretrained(args.output, torch_dtype=torch.bfloat16)
    n_reload = sum(x.numel() for x in reloaded.parameters()) / 1e9
    assert abs(n_reload - n_total) < 1e-6, f"参数量不一致: {n_reload} vs {n_total}"
    print(f"[verify] reload OK, params={n_reload:.2f}B")


if __name__ == "__main__":
    main()
