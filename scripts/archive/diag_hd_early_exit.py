#!/usr/bin/env python3
"""
diag_hd_early_exit.py — 诊断 HD 早退为何未生效 (pod 上跑, ~1 分钟)。

背景: E2E bench 里 --hd-early-k 16 后 t_vit_hd 仍是全深度耗时 (2016: 167ms vs
纯 ViT bench k=16 的 84ms), 但纯 ViT bench 的同款截断技巧明明有效。
本脚本用 forward hook 数"每次调用实际执行了几个 vision block", 区分三种可能:
  A. 运行时导入的 modeling 文件不是 CFS 仓库这份 (打印 __file__)
  B. _hd_vit_truncated 读到的 k 不对 (config 字典 identity 问题)
  C. blocks 截断赋值本身无效 (transformers 行为)

用法 (pod):
  python scripts/diag_hd_early_exit.py \
      --base-model /workspace/model_cache/Qwen2.5-VL-3B-Instruct
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import time

import torch

DTYPE = torch.bfloat16
DEVICE = "cuda"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model",
                   default=os.path.join(
                       os.environ.get("MODEL_CACHE", "/workspace/model_cache"),
                       "Qwen2.5-VL-3B-Instruct"))
    p.add_argument("--R", type=int, default=2016)
    args = p.parse_args()

    from transformers import AutoConfig
    import llava.model.language_model.modeling_qwen2_5vl_dat as M

    print(f"[A] modeling file : {M.__file__}")
    print(f"[A] has truncated : {hasattr(M.Qwen2_5_VLDATForConditionalGeneration, '_hd_vit_truncated')}")

    # ── 构建模型 (与 test_inference_bench 的 convert 路径一致) ──────────────
    base_cfg = AutoConfig.from_pretrained(args.base_model)
    n_layers = getattr(base_cfg, 'num_hidden_layers', None) or \
               base_cfg.text_config.num_hidden_layers
    dat_extra_args = {
        'grid_size': 20, 'off_ksize': 3, 'off_grps': 8, 'inter_size': 128,
        'hr_scale': 3, 'hd_proj': True, 'use_intention_branch': True,
        'intention_as_gate': True, 'use_spatial_attn_guide': False,
        'hd_gate_init': None, 'hd_gate_freeze': False,
        'inject_lr_image': False, 'image_hd_for_question': False,
        'use_fused_vit': False, 'use_shared_vit': False,
        'hd_early_exit_k': 0,
        'layers': ''.join('D' if i % 6 == 0 else 'L' for i in range(n_layers)),
    }
    print(f"[.] building DAT from base …")
    model = M.convert_qwen2_5vl_to_dat(
        args.base_model, dat_extra_args, torch_dtype=DTYPE,
    ).to(DEVICE).eval()

    ea = model.config.dat_extra_args
    print(f"[B] dict identity : passed-in={id(dat_extra_args)}  model.config={id(ea)}  same={ea is dat_extra_args}")

    # ── hook: 数每次 forward 实际执行的 block ───────────────────────────────
    visual = model.model.visual
    executed = []
    handles = [blk.register_forward_hook(lambda m, a, o, _i=i: executed.append(_i))
               for i, blk in enumerate(visual.blocks)]

    ps = model.config.vision_config.patch_size
    tps = model.config.vision_config.temporal_patch_size
    h = w = args.R // ps
    dim = 3 * tps * ps * ps
    pv = torch.randn(h * w, dim, device=DEVICE, dtype=DTYPE)
    thw = torch.tensor([[1, h, w]], device=DEVICE, dtype=torch.long)

    def run_once(tag):
        executed.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model._generate_hd_features(pv, thw)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        k_in_cfg = model.config.dat_extra_args.get('hd_early_exit_k', 'MISSING')
        print(f"[C] {tag:<18} cfg_k={k_in_cfg!s:<8} blocks_executed={len(executed)}"
              f"  (ids: {executed[:3]}…{executed[-3:] if executed else []})  t={ms:7.1f}ms")

    # warmup
    model._generate_hd_features(pv, thw)

    run_once("k=0 (off)")

    ea['hd_early_exit_k'] = 16
    run_once("k=16 via ea ref")

    # 直接从 model.config 再设一遍 (排除 ea 引用歧义)
    model.config.dat_extra_args['hd_early_exit_k'] = 8
    run_once("k=8 via config")

    # 直接手动截断 (绕过 config, 验证机制本身)
    ea['hd_early_exit_k'] = 0
    full = visual.blocks
    visual.blocks = full[:4]
    run_once("manual blocks[:4]")
    visual.blocks = full

    for hdl in handles:
        hdl.remove()
    print("[DONE]")


if __name__ == "__main__":
    main()
