#!/usr/bin/env python3
"""复现并定位 DAT 类 base 路径 (无 HD 输入) 的显存异常。

sysbench 观察: batch_decode R=2016 B=8 时 peak mem
  qwen 原生 22.7GB / DAT-base 29.2GB / DAT-hd 23.4GB
DAT-base 与 qwen 计算内容应当完全相同, 多出的 ~6.5GB 是异常。

实验设计 (R=2016, B=8, 合成图):
  1) 三条路径各测 forward(use_cache=True) 与 generate(T=8) 的峰值显存
  2) 对 DAT-base forward 挂逐层 hook 测局部峰值, 定位多吃显存的层
     (若集中在 6 个 DAT 层 → 无 HD fallback 路径的问题)

用法 (pod):
  python scripts/debug_base_mem.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse

import torch
from PIL import Image

DTYPE = torch.bfloat16
DEVICE = "cuda"
GB = 1024 ** 3


def peak_gb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / GB


def main():
    p = argparse.ArgumentParser()
    _cache = os.environ.get("MODEL_CACHE", "/workspace/model_cache")
    p.add_argument("--base-model", default=os.path.join(_cache, "Qwen2.5-VL-3B-Instruct"))
    p.add_argument("--dat-ckpt", default=os.path.join(_cache, "Qwen2.5-VL-3B-Instruct-DAT-rand"))
    p.add_argument("--R", type=int, default=2016)
    p.add_argument("--B", type=int, default=8)
    args = p.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from llava.model.language_model.modeling_qwen2_5vl_dat import (
        Qwen2_5_VLDATForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.tokenizer.padding_side = "left"

    print(f"[load] qwen native + DAT ckpt  (R={args.R}, B={args.B})")
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=DTYPE, device_map={"": 0}).eval()
    dat = Qwen2_5_VLDATForConditionalGeneration.from_pretrained(
        args.dat_ckpt, torch_dtype=DTYPE, device_map={"": 0}).eval()
    print(f"[info] qwen attn_impl={qwen.config._attn_implementation}"
          f"  dat attn_impl={dat.config._attn_implementation}")
    print(f"[info] static mem after load: {torch.cuda.memory_allocated()/GB:.2f} GB")

    # ── 输入 (base 路径: 满分辨率 + 完整文本) ──────────────────────────────
    img = Image.new("RGB", (args.R, args.R), color=(100, 149, 237))
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image in detail."},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = processor(images=[img] * args.B, text=[text] * args.B, return_tensors="pt",
                    padding=True, min_pixels=args.R ** 2, max_pixels=args.R ** 2)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    pv = inp["pixel_values"].to(DTYPE)
    print(f"[info] seq_len={inp['input_ids'].shape[1]}")

    common = dict(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"),
                  pixel_values=pv, image_grid_thw=inp["image_grid_thw"])
    gen_kw = dict(max_new_tokens=8, min_new_tokens=8, do_sample=False,
                  use_cache=True, pad_token_id=processor.tokenizer.eos_token_id)

    # ── 1) 三路径峰值 ──────────────────────────────────────────────────────
    print("\n== 峰值显存 (GB) ==")
    for name, model, extra in [
        ("qwen     ", qwen, {}),
        ("dat-base ", dat, dict(pixel_values_hd=None, image_grid_thw_hd=None)),
    ]:
        m_fwd = peak_gb(lambda: model(**common, **extra, use_cache=True))
        m_gen = peak_gb(lambda: model.generate(**common, **extra, **gen_kw))
        print(f"  {name} forward={m_fwd:6.2f}   generate(T=8)={m_gen:6.2f}")

    # ── 2) DAT-base 逐层局部峰值定位 ───────────────────────────────────────
    print("\n== DAT-base 逐层局部峰值 (forward, use_cache=True) ==")
    layers = dat.model.language_model.layers
    dat_idx = {i for i, l in enumerate(layers)
               if "DAT" in type(l).__name__ or hasattr(l, "hd_input_layernorm")
               or "DAT" in type(getattr(l, "self_attn", l)).__name__}
    records = {}
    handles = []

    def mk_pre(i):
        def pre(_m, _args, _kw):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            records[i] = {"before": torch.cuda.memory_allocated() / GB}
        return pre

    def mk_post(i):
        def post(_m, _args, _kw, _out):
            torch.cuda.synchronize()
            records[i]["peak"] = torch.cuda.max_memory_allocated() / GB
            records[i]["after"] = torch.cuda.memory_allocated() / GB
        return post

    for i, l in enumerate(layers):
        handles.append(l.register_forward_pre_hook(mk_pre(i), with_kwargs=True))
        handles.append(l.register_forward_hook(mk_post(i), with_kwargs=True))
    try:
        with torch.no_grad():
            dat(**common, pixel_values_hd=None, image_grid_thw_hd=None, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    rows = sorted(records.items(),
                  key=lambda kv: kv[1]["peak"] - kv[1]["before"], reverse=True)
    print(f"  {'layer':>5} {'type':>4} {'peak-before(GB)':>16} {'after-before(GB)':>17}")
    for i, r in rows[:10]:
        tag = "DAT" if i in dat_idx else "L"
        print(f"  {i:>5} {tag:>4} {r['peak'] - r['before']:>16.3f} {r['after'] - r['before']:>17.3f}")
    d_avg = sum(r["peak"] - r["before"] for i, r in records.items() if i in dat_idx) / max(1, len(dat_idx))
    l_avg = sum(r["peak"] - r["before"] for i, r in records.items() if i not in dat_idx) / max(1, len(records) - len(dat_idx))
    print(f"\n  DAT 层平均局部峰值增量: {d_avg:.3f} GB  (x{len(dat_idx)})")
    print(f"  L   层平均局部峰值增量: {l_avg:.3f} GB  (x{len(records) - len(dat_idx)})")


if __name__ == "__main__":
    main()
