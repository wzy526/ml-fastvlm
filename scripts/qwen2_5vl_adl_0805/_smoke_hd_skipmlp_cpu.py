#!/usr/bin/env python3
"""CPU smoke test for hd_skip_merger_mlp (no GPU on this box; shell capped at 2GB).

Builds a TINY random-init Qwen2.5-VL (real architecture, shrunk dims: text
hidden 128, vision hidden 64 -> hd_feat_dim 64*4=256), saves it, then loads it
through the same convert_qwen2_5vl_to_dat + from_pretrained path training uses.

Checks:
  1. hd_feat_dim injection: k_proj_hd/v_proj_hd in_features == vision_hidden*4,
     hd_input_layernorm sized to match
  2. _generate_hd_features returns [H, W, hd_feat_dim] maps (merger MLP
     skipped) and merger.mlp is restored afterwards
  3. full forward with LR + HD inputs produces finite logits
  4. backward: k_proj_hd / v_proj_hd / conv_off_proj receive finite grads (the
     DAT sampling path stays differentiable with the wider channels)

Run:
    cd /root/autodl-tmp/ml-fastvlm && python scripts/qwen2_5vl_adl_0805/_smoke_hd_skipmlp_cpu.py
"""
import importlib
import os
import sys
import tempfile
import types

if os.environ.get("SMOKE_BLOCK_HEAVY"):
    # The dev sandbox has ~1GB headroom; transformers drags in scipy / pandas /
    # pyarrow / torchvision through optional integrations. Block them (they are
    # all guarded by is_*_available checks).
    _BLOCKED = ("scipy", "pandas", "pyarrow", "datasets", "torchvision",
                "deepspeed", "peft", "timm", "cv2")

    class _HeavyBlocker:
        def find_spec(self, name, path=None, target=None):
            root = name.split('.')[0]
            if root in _BLOCKED:
                raise ImportError(f"blocked heavy module: {name}")
            return None

    sys.meta_path.insert(0, _HeavyBlocker())

# transformers probes deepspeed during save/load; the import costs ~100MB we
# don't have. None in sys.modules makes ``import deepspeed`` raise ImportError,
# which every transformers call site guards.
sys.modules['deepspeed'] = None

import torch
from PIL import Image

torch.set_num_threads(1)

if os.environ.get("SMOKE_RSS_WATCH"):
    import threading
    import time

    def _watch():
        while True:
            rss_kb = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0])
            print(f"      [rss] {rss_kb // 1024} MB", flush=True)
            time.sleep(1)

    threading.Thread(target=_watch, daemon=True).start()

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

# The dev sandbox caps this process at 2GB; ``import llava`` pulls the full
# training stack (deepspeed etc.) and OOMs. modeling_qwen2_5vl_dat itself only
# needs torch/transformers/einops, so stub the parent packages and import the
# module directly.
for _name, _sub in [("llava", "llava"),
                    ("llava.model", "llava/model"),
                    ("llava.model.language_model", "llava/model/language_model")]:
    _m = types.ModuleType(_name)
    _m.__path__ = [os.path.join(REPO, _sub)]
    sys.modules[_name] = _m

MODEL_PATH = "/root/autodl-tmp/models_data/Qwen2.5-VL-3B-Instruct"
IMAGE = "/root/autodl-tmp/models_data/sft_data/train_split/chartqa/chartqa_003324.jpg"
FACTOR = 28


def build_tiny_base(tmpdir):
    from transformers import Qwen2_5_VLConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    cfg = Qwen2_5_VLConfig.from_pretrained(MODEL_PATH)
    d = cfg.to_dict()
    # Shrink the vocab too (151936 x hidden embeddings dominate memory);
    # remap the vision special-token ids into the tiny vocab.
    d['text_config'].update(dict(
        hidden_size=128, intermediate_size=256, num_hidden_layers=6,
        num_attention_heads=8, num_key_value_heads=4,
        layer_types=['full_attention'] * 6, max_window_layers=6,
        vocab_size=1024,
    ))
    for key, val in dict(image_token_id=900, video_token_id=903,
                         vision_start_token_id=901, vision_end_token_id=902,
                         vision_token_id=904).items():
        if key in d:
            d[key] = val
    d['text_config']['bos_token_id'] = 0
    d['text_config']['eos_token_id'] = 1
    d['text_config']['rope_scaling'] = dict(
        d['text_config'].get('rope_scaling') or {'type': 'mrope'})
    # head_dim = 128/8 = 16 -> mrope sections must sum to head_dim/2 = 8
    d['text_config']['rope_scaling']['mrope_section'] = [2, 3, 3]
    d['vision_config'].update(dict(
        depth=4, hidden_size=64, num_heads=4, intermediate_size=128,
        out_hidden_size=128, fullatt_block_indexes=[3],
    ))
    tiny_cfg = Qwen2_5_VLConfig(**d)
    torch.manual_seed(0)
    base = Qwen2_5_VLForConditionalGeneration(tiny_cfg)
    base = base.float()
    base.save_pretrained(tmpdir)
    return tiny_cfg


def main():
    M = importlib.import_module("llava.model.language_model.modeling_qwen2_5vl_dat")
    convert_qwen2_5vl_to_dat = M.convert_qwen2_5vl_to_dat
    Qwen2_5_VLAttentionDAT = M.Qwen2_5_VLAttentionDAT

    if not torch.cuda.is_available():
        # flash_attn has no CPU kernels; swap in math-equivalent naive
        # attention-with-LSE so the LSE merge path runs on CPU.
        def _cpu_attn_lse(q, k, v, causal=False):
            d = q.shape[-1]
            scores = (q.float() @ k.float().transpose(-1, -2)) / d ** 0.5
            if causal:
                nq, nk = scores.shape[-2:]
                mask = torch.ones(nq, nk, dtype=torch.bool).triu(nk - nq + 1)
                scores = scores.masked_fill(mask, float('-inf'))
            lse = torch.logsumexp(scores, dim=-1)              # [B,H,Nq]
            out = torch.softmax(scores, dim=-1) @ v.float()
            return out.transpose(1, 2).to(q.dtype), lse        # [B,Nq,H,D]

        def _cpu_cross_varlen(q_list, k_list, v_list):
            outs, lses = [], []
            for q, k, v in zip(q_list, k_list, v_list):
                o, l = _cpu_attn_lse(q.transpose(0, 1)[None],
                                     k.transpose(0, 1)[None],
                                     v.transpose(0, 1)[None], causal=False)
                outs.append(o)
                lses.append(l)
            return outs, lses

        M._dat_attn_with_lse = _cpu_attn_lse
        M._dat_cross_attn_varlen = _cpu_cross_varlen
        print("      [cpu] patched LSE attention with naive CPU fallback")
    # NOT AutoProcessor: loading the Qwen tokenizer costs several hundred MB
    # (OOM in the 2GB sandbox). The image processor alone is cheap and we
    # assemble input_ids by hand from the known special-token ids.
    from transformers import AutoImageProcessor

    dat_extra_args = {
        'grid_size': 20,
        'off_ksize': 3,
        'off_grps': 8,
        'inter_size': 128,
        'hr_scale': 3,
        'hd_proj': True,
        'layers': 'DLLLLL',
        'use_intention_branch': True,
        'intention_as_gate': True,
        'use_spatial_attn_guide': False,
        'hd_gate_init': None,
        'hd_gate_freeze': False,
        'use_fused_vit': False,
        'use_shared_vit': False,
        'image_hd_for_question': False,
        'inject_lr_image': False,
        'hd_early_exit_k': 0,
        'hd_skip_merger_mlp': True,
    }

    # /tmp is tmpfs (charged against the cgroup memory cap) — use real disk.
    tmpdir = tempfile.mkdtemp(prefix="tiny_qwen25vl_", dir="/root/autodl-tmp")
    print(f"[1/4] building tiny random-init Qwen2.5-VL at {tmpdir} ...")
    build_tiny_base(tmpdir)

    print("      converting to DAT via the training path (from_pretrained) ...")
    model = convert_qwen2_5vl_to_dat(tmpdir, dat_extra_args, torch_dtype=torch.float32)
    model.eval()

    vis_cfg = model.config.vision_config
    expect_dim = vis_cfg.hidden_size * vis_cfg.spatial_merge_size ** 2
    print(f"      expected hd_feat_dim = {expect_dim}")
    assert model.config.dat_extra_args.get('hd_feat_dim') == expect_dim

    dat_layers = [l.self_attn for l in model.model.language_model.layers
                  if isinstance(l.self_attn, Qwen2_5_VLAttentionDAT)]
    assert len(dat_layers) == 1, len(dat_layers)
    for a in dat_layers:
        assert a.k_proj_hd.weight.shape[1] == expect_dim, a.k_proj_hd.weight.shape
        assert a.v_proj_hd.weight.shape[1] == expect_dim
        assert a.hd_input_layernorm.weight.numel() == expect_dim
        assert a.hd_off_dim == expect_dim // a.off_grps
    print(f"      OK: k/v_proj_hd + hd_input_layernorm sized to {expect_dim}")

    # --- inputs: small LR + ~4x HD ---
    img = Image.open(IMAGE).convert("RGB")
    ip = AutoImageProcessor.from_pretrained(MODEL_PATH)
    lr_inputs = ip(images=[img], return_tensors="pt",
                   min_pixels=56 * 56, max_pixels=112 * 112)
    lr_thw = lr_inputs["image_grid_thw"][0]
    lr_px = int(lr_thw[1]) * int(lr_thw[2]) * FACTOR * FACTOR
    hd_target = min(lr_px * 4, img.width * img.height)
    hr_inputs = ip(images=[img], return_tensors="pt",
                   min_pixels=hd_target, max_pixels=hd_target)
    print(f"      LR thw={lr_thw.tolist()} HD thw={hr_inputs['image_grid_thw'][0].tolist()}")

    # Hand-rolled input_ids (avoids loading the tokenizer): a few "text" ids,
    # then <|vision_start|> + N x <|image_pad|> + <|vision_end|>, then a
    # question-ish tail and an answer-ish tail so DAT has answer positions.
    tcfg = model.config
    merge = model.config.vision_config.spatial_merge_size
    n_img = int(lr_thw[0] * lr_thw[1] * lr_thw[2]) // (merge ** 2)
    # DAT locates the intention token via the <|im_start|> module constant
    # (151644 — outside the tiny vocab), so remap it for this test.
    M.IM_START_TOKEN_ID = 950
    ids = ([100, 200, 300]
           + [tcfg.vision_start_token_id]
           + [tcfg.image_token_id] * n_img
           + [tcfg.vision_end_token_id]
           + list(range(400, 412))          # "question"
           + [M.IM_START_TOKEN_ID]
           + list(range(500, 512)))         # "answer"
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    print("[2/4] _generate_hd_features dim check ...")
    with torch.no_grad():
        feats = model._generate_hd_features(
            hr_inputs["pixel_values"].float(), hr_inputs["image_grid_thw"])
    assert feats[0].shape[-1] == expect_dim, feats[0].shape
    assert not isinstance(model.model.visual.merger.mlp, torch.nn.Identity), \
        "merger.mlp not restored after context manager"
    print(f"      OK: HD feature map {tuple(feats[0].shape)}; merger.mlp restored")

    print("[3/4] full forward with LR + HD ...")
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=lr_inputs["pixel_values"].float(),
        image_grid_thw=lr_inputs["image_grid_thw"],
        pixel_values_hd=hr_inputs["pixel_values"].float(),
        image_grid_thw_hd=hr_inputs["image_grid_thw"],
        use_cache=False,
    )
    assert torch.isfinite(out.logits).all(), "non-finite logits"
    print(f"      OK: logits {tuple(out.logits.shape)} finite")

    print("[4/4] backward through DAT params ...")
    loss = out.logits[:, -8:].float().pow(2).mean()
    loss.backward()
    a = dat_layers[0]
    for name in ("k_proj_hd", "v_proj_hd", "conv_off_proj"):
        g = getattr(a, name).weight.grad
        assert g is not None and torch.isfinite(g).all(), f"{name}: bad grad"
        print(f"      {name}.weight.grad norm = {g.norm().item():.3e}")

    print("\nALL CHECKS PASSED")


if __name__ == '__main__':
    main()
