#!/usr/bin/env python3
#
# Qwen2-VL Instruct TTFT 和 FLOPs 测试脚本
# 使用 Qwen2-VL 原生参数（动态分辨率、native processor）
#
import os
import sys
import time
import json
import argparse
import subprocess

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# ──────────────────────────────────────────────────────────────
# Qwen2-VL 原生参数
# ──────────────────────────────────────────────────────────────
# patch_size=14, merge_size=2 → 每边 patch 数 = H / (14*2)
# 336×336 → 12×12 = 144 visual tokens (after merger)
# 动态分辨率：Qwen2-VL-2B 默认 min_pixels=256*28*28, max_pixels=1280*28*28
# Qwen2-VL-7B 默认 min_pixels=256*28*28, max_pixels=16384*28*28
PATCH_SIZE = 14
MERGE_SIZE = 2


def visual_tokens_from_grid(h_patches: int, w_patches: int) -> int:
    """根据 Qwen2-VL 的 PatchMerger 计算 visual token 数量。"""
    return (h_patches // MERGE_SIZE) * (w_patches // MERGE_SIZE)


def visual_tokens_from_image(image: Image.Image, processor) -> int:
    """用 processor 实际处理一张图，返回 visual token 数。"""
    inputs = processor(images=[image], return_tensors="pt")
    thw = inputs["image_grid_thw"][0]  # (T, H, W) in patches
    t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
    return visual_tokens_from_grid(h, w) * t


# ──────────────────────────────────────────────────────────────
# Dataset  (支持 GQA / COCO captions)
# ──────────────────────────────────────────────────────────────

def _detect_dataset_type(raw: dict | list) -> str:
    """根据 JSON 结构推断数据集类型：'coco' 或 'gqa'。"""
    if isinstance(raw, dict):
        # COCO captions format: {"info":..., "images":[...], "annotations":[...]}
        if "images" in raw and "annotations" in raw:
            return "coco"
        # GQA dict format: {qid: {question, imageId, ...}}
        return "gqa"
    # list format → GQA list
    return "gqa"


class Qwen2VLTestDataset(Dataset):
    """Dataset wrapper for Qwen2-VL TTFT testing.

    支持：
    - GQA JSON（dict 或 list 格式）
    - COCO captions JSON（captions_val2017.json 等）
    - 纯图像目录（data_path=None，直接扫描 image_folder）
    """

    COCO_QUESTION = "Describe this image briefly."

    def __init__(self, data_path, image_folder, max_samples=None):
        self.image_folder = image_folder
        self.samples = []

        # ── 纯图像目录模式 ──
        if data_path is None:
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            files = sorted(
                f for f in os.listdir(image_folder)
                if os.path.splitext(f)[1].lower() in exts
            )
            for i, fname in enumerate(files):
                if max_samples and i >= max_samples:
                    break
                self.samples.append({
                    "qid": f"img_{i}",
                    "question": self.COCO_QUESTION,
                    "file_name": fname,
                    "answer": "",
                })
            return

        with open(data_path, "r") as f:
            raw = json.load(f)

        ds_type = _detect_dataset_type(raw)

        if ds_type == "coco":
            # COCO captions format
            # 建立 image_id → file_name 映射
            id2fname = {img["id"]: img["file_name"] for img in raw["images"]}
            seen_ids = set()
            for ann in raw["annotations"]:
                if max_samples and len(self.samples) >= max_samples:
                    break
                img_id = ann["image_id"]
                if img_id in seen_ids:
                    continue
                seen_ids.add(img_id)
                self.samples.append({
                    "qid": str(ann["id"]),
                    "question": self.COCO_QUESTION,
                    "file_name": id2fname.get(img_id, f"{img_id:012d}.jpg"),
                    "answer": ann.get("caption", ""),
                })
        elif isinstance(raw, dict):
            # GQA dict format
            for qid, item in raw.items():
                if max_samples and len(self.samples) >= max_samples:
                    break
                self.samples.append({
                    "qid": qid,
                    "question": item.get("question", ""),
                    "file_name": f"{item.get('imageId', qid)}.jpg",
                    "answer": item.get("answer", ""),
                })
        else:
            # GQA list format
            for i, item in enumerate(raw):
                if max_samples and len(self.samples) >= max_samples:
                    break
                self.samples.append({
                    "qid": item.get("qid", f"qid_{i}"),
                    "question": item.get("question", ""),
                    "file_name": f"{item.get('imageId', i)}.jpg",
                    "answer": item.get("answer", ""),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image_path = os.path.join(self.image_folder, s["file_name"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (336, 336), color="gray")
        return {"qid": s["qid"], "question": s["question"], "image": image, "answer": s["answer"]}


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

def load_qwen2vl(model_path: str, min_pixels: int = None, max_pixels: int = None):
    """加载 Qwen2-VL Instruct 模型和 processor。

    min/max_pixels 控制动态分辨率范围，None 则使用模型默认值。
    """
    processor_kwargs = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels

    print(f"Loading Qwen2-VL from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    # 强制单 GPU 加载，避免多 GPU device_map="auto" 的跨设备索引问题
    # Qwen2-VL-2B 约 4GB bfloat16，单 GPU 完全可容纳
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    model_device = next(model.parameters()).device
    print(f"Model loaded. dtype={next(model.parameters()).dtype}, device={model_device}")
    return model, processor


# ──────────────────────────────────────────────────────────────
# TTFT
# ──────────────────────────────────────────────────────────────

def measure_ttft(model, processor, sample: dict) -> tuple[float, int]:
    """返回 (ttft_ms, visual_token_count)。"""
    image = sample["image"]
    question = sample["question"]

    # apply_chat_template 无法正确处理 PIL Image 对象（会返回空串）
    # 改为手动构建 Qwen2-VL 标准 chat 格式，让 processor 负责展开 image_pad token
    text_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # 预处理（不计入 TTFT）
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    # device_map={"": 0} 确保所有参数都在 cuda:0，将输入统一移到同一设备
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    inputs_fixed = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            inputs_fixed[k] = v
        elif k in ('input_ids', 'attention_mask', 'image_grid_thw'):
            inputs_fixed[k] = v.to(device=model_device, dtype=torch.long)
        elif k == 'pixel_values':
            inputs_fixed[k] = v.to(device=model_device, dtype=model_dtype)
        else:
            inputs_fixed[k] = v.to(device=model_device)
    inputs = inputs_fixed

    # 计算 visual token 数量
    thw = inputs["image_grid_thw"][0]
    t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
    n_visual = visual_tokens_from_grid(h, w) * t

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = time.time()
    with torch.inference_mode():
        # 直接 forward pass 代替 generate()：
        # Qwen2-VL 的 generate() 存在 cache_position 追踪问题（transformers 兼容性 bug）
        # prefill 的 forward pass 时间即为 TTFT（生成第一个 token 所需时间）
        outputs = model(**inputs, use_cache=False)
        # 从最后一个位置取 argmax 即第一个生成 token（保证完整走完 lm_head）
        _ = outputs.logits[:, -1, :].argmax(dim=-1)
    torch.cuda.synchronize()
    ttft_ms = (time.time() - start) * 1000

    return ttft_ms, n_visual


def run_ttft_benchmark(model_path: str, data_path: str, image_folder: str,
                       max_samples: int = 200,
                       min_pixels: int = None, max_pixels: int = None,
                       output_file: str = None, warmup: int = 3):
    """在 GQA 数据集上跑 TTFT benchmark，返回结果 dict。"""
    model, processor = load_qwen2vl(model_path, min_pixels, max_pixels)
    dataset = Qwen2VLTestDataset(
        data_path if data_path and data_path != "none" else None,
        image_folder,
        max_samples=max_samples,
    )

    print(f"Dataset size: {len(dataset)}")

    # warmup
    print(f"Warmup ({warmup} steps)...")
    for i in range(min(warmup, len(dataset))):
        measure_ttft(model, processor, dataset[i])

    ttft_list, visual_token_list = [], []
    for i in tqdm(range(len(dataset)), desc="TTFT"):
        try:
            ttft, n_vis = measure_ttft(model, processor, dataset[i])
            ttft_list.append(ttft)
            visual_token_list.append(n_vis)
        except Exception as e:
            print(f"[WARN] sample {i} failed: {e}")

    results = {
        "model_path": model_path,
        "num_samples": len(ttft_list),
        "avg_ttft_ms": float(np.mean(ttft_list)) if ttft_list else 0,
        "median_ttft_ms": float(np.median(ttft_list)) if ttft_list else 0,
        "p95_ttft_ms": float(np.percentile(ttft_list, 95)) if ttft_list else 0,
        "min_ttft_ms": float(np.min(ttft_list)) if ttft_list else 0,
        "max_ttft_ms": float(np.max(ttft_list)) if ttft_list else 0,
        "avg_visual_tokens": float(np.mean(visual_token_list)) if visual_token_list else 0,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n===== TTFT Results =====")
    print(f"  Samples:          {results['num_samples']}")
    print(f"  Avg TTFT:         {results['avg_ttft_ms']:.1f} ms")
    print(f"  Median TTFT:      {results['median_ttft_ms']:.1f} ms")
    print(f"  P95 TTFT:         {results['p95_ttft_ms']:.1f} ms")
    print(f"  Avg visual tokens:{results['avg_visual_tokens']:.1f}")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {output_file}")

    return results


# ──────────────────────────────────────────────────────────────
# FLOPs
# ──────────────────────────────────────────────────────────────

def compute_llm_flops_manual(config, total_seq_len: int) -> int:
    """手动计算 Qwen2 LLM 的 prefill FLOPs（乘法次数）。

    公式与 flops_test.py 保持一致。
    """
    H = config.hidden_size
    L = getattr(config, "num_hidden_layers", 28)
    I = getattr(config, "intermediate_size", 18944)
    Nh = config.num_attention_heads
    Nkv = getattr(config, "num_key_value_heads", Nh)
    head_dim = H // Nh
    kv_dim = head_dim * Nkv
    S = total_seq_len

    # Attention per layer
    q_flops = H * H * S
    k_flops = H * kv_dim * S
    v_flops = H * kv_dim * S
    attn_flops = S * S * head_dim * Nh * 2  # QK + AV
    o_flops = H * H * S
    attn_total = q_flops + k_flops + v_flops + attn_flops + o_flops

    # FFN (SwiGLU) per layer
    gate = H * I * S
    up = H * I * S
    act = I * S
    down = I * H * S
    ffn_total = gate + up + act + down

    # Norm per layer (2× RMSNorm)
    norm_total = H * S * 2

    layer_flops = attn_total + ffn_total + norm_total

    # lm_head + final norm
    vocab_size = getattr(config, "vocab_size", 152064)
    lm_head_flops = H * vocab_size * S
    final_norm = H * S

    return L * layer_flops + lm_head_flops + final_norm


def compute_vision_flops_manual(h_patches: int, w_patches: int,
                                vision_config) -> int:
    """手动估算 Qwen2-VL vision encoder FLOPs（transformer blocks + merger）。

    只做粗略估算：ViT-style blocks + MLP merger。
    """
    embed_dim = getattr(vision_config, "embed_dim", 1280)
    depth = getattr(vision_config, "depth", 32)
    num_heads = getattr(vision_config, "num_heads", 16)
    mlp_ratio = getattr(vision_config, "mlp_ratio", 4.0)
    hidden_size = getattr(vision_config, "hidden_size", 1536)

    n_tokens = h_patches * w_patches
    head_dim = embed_dim // num_heads

    # Per ViT block
    attn = n_tokens * n_tokens * head_dim * num_heads * 2 + n_tokens * embed_dim * embed_dim * 3
    ffn = n_tokens * embed_dim * int(embed_dim * mlp_ratio) * 2
    block_flops = attn + ffn

    # Merger MLP: [n_tokens, embed_dim * merge_size^2] → [n_tokens_merged, hidden_size]
    merge_in = embed_dim * (MERGE_SIZE ** 2)
    n_merged = (h_patches // MERGE_SIZE) * (w_patches // MERGE_SIZE)
    merger_flops = n_merged * merge_in * hidden_size * 2  # two linear layers approx

    return depth * block_flops + merger_flops


def run_flops_benchmark(model_path: str,
                        min_pixels: int = None, max_pixels: int = None,
                        sample_image: Image.Image = None,
                        text_len: int = 30,
                        output_file: str = None):
    """计算 Qwen2-VL 的 FLOPs。

    拆分为：vision encoder + LLM（prefill）。
    """
    if sample_image is None:
        sample_image = Image.new("RGB", (336, 336), color="gray")

    model, processor = load_qwen2vl(model_path, min_pixels, max_pixels)

    # 获取真实 grid_thw（需要提供 text 让 processor 正确展开 image token）
    _dummy_text = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = processor(text=[_dummy_text], images=[sample_image], return_tensors="pt")
    thw = inputs["image_grid_thw"][0]
    t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
    n_visual = visual_tokens_from_grid(h, w) * t
    total_seq = n_visual + text_len

    print(f"\n===== FLOPs Estimation =====")
    print(f"  image grid (T,H,W) = ({t},{h},{w}) patches")
    print(f"  visual tokens after merger = {n_visual}")
    print(f"  text tokens (assumed) = {text_len}")
    print(f"  total LLM sequence length = {total_seq}")

    # Vision FLOPs (manual)
    vision_cfg = model.config.vision_config
    vision_flops = compute_vision_flops_manual(h, w, vision_cfg) * t
    print(f"  Vision encoder FLOPs (manual): {vision_flops:,.0f}  ({vision_flops/1e9:.2f} G)")

    # LLM FLOPs (manual)
    llm_flops = compute_llm_flops_manual(model.config, total_seq)
    print(f"  LLM FLOPs (manual):            {llm_flops:,.0f}  ({llm_flops/1e9:.2f} G)")

    total_flops = vision_flops + llm_flops
    print(f"  Total FLOPs:                   {total_flops:,.0f}  ({total_flops/1e9:.2f} G)")

    # fvcore vision encoder
    fvcore_vision = 0
    if FVCORE_AVAILABLE:
        try:
            print("\n  [fvcore] Computing vision encoder FLOPs...")
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            pv = inputs["pixel_values"].to(device=device, dtype=dtype)
            gt = inputs["image_grid_thw"].to(device=device)
            analyzer = FlopCountAnalysis(model.visual, (pv, gt))
            fvcore_vision = analyzer.total()
            print(f"  [fvcore] Vision encoder FLOPs: {fvcore_vision:,.0f}  ({fvcore_vision/1e9:.2f} G)")
        except Exception as e:
            print(f"  [fvcore] Vision FLOPs failed: {e}")

    results = {
        "model_path": model_path,
        "image_grid_thw": [t, h, w],
        "n_visual_tokens": n_visual,
        "text_tokens": text_len,
        "total_seq_len": total_seq,
        "vision_flops_manual": vision_flops,
        "vision_flops_fvcore": fvcore_vision,
        "llm_flops_manual": llm_flops,
        "total_flops": total_flops,
        "total_flops_G": total_flops / 1e9,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {output_file}")

    return results


# ──────────────────────────────────────────────────────────────
# Comprehensive test (mirrors test_dat_llava1_5_ttft_flops.py)
# ──────────────────────────────────────────────────────────────

def run_comprehensive_test(args):
    print("=" * 80)
    print("Qwen2-VL Instruct  综合测试  (TTFT + FLOPs)")
    print("=" * 80)
    print(f"  model:      {args.model_path}")
    print(f"  data:       {args.data_path}")
    print(f"  images:     {args.image_folder}")
    print(f"  min_pixels: {args.min_pixels}")
    print(f"  max_pixels: {args.max_pixels}")
    print(f"  max_samples:{args.max_samples}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    ttft_out = f"ttft_qwen2vl_{ts}.json"
    flops_out = f"flops_qwen2vl_{ts}.json"

    # ── TTFT ──
    print("\n" + "=" * 60)
    print("1. TTFT benchmark")
    print("=" * 60)
    ttft_results = run_ttft_benchmark(
        model_path=args.model_path,
        data_path=args.data_path if args.data_path and args.data_path != "none" else None,
        image_folder=args.image_folder,
        max_samples=args.max_samples,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        output_file=ttft_out,
        warmup=args.warmup,
    )

    # ── FLOPs ──
    print("\n" + "=" * 60)
    print("2. FLOPs estimation")
    print("=" * 60)

    # 用数据集里第一张图做 FLOPs 估计（代表性样本）
    try:
        sample_ds = Qwen2VLTestDataset(
            args.data_path if args.data_path != "none" else None,
            args.image_folder,
            max_samples=1,
        )
        sample_image = sample_ds[0]["image"]
    except Exception:
        sample_image = None

    flops_results = run_flops_benchmark(
        model_path=args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        sample_image=sample_image,
        text_len=args.text_len,
        output_file=flops_out,
    )

    # ── Summary ──
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Avg TTFT:              {ttft_results['avg_ttft_ms']:.1f} ms")
    print(f"  Median TTFT:           {ttft_results['median_ttft_ms']:.1f} ms")
    print(f"  P95 TTFT:              {ttft_results['p95_ttft_ms']:.1f} ms")
    print(f"  Avg visual tokens:     {ttft_results['avg_visual_tokens']:.1f}")
    print(f"  Total FLOPs (1 image): {flops_results['total_flops_G']:.2f} G")
    print(f"  Vision FLOPs:          {flops_results['vision_flops_manual']/1e9:.2f} G  (manual)")
    print(f"  LLM FLOPs:             {flops_results['llm_flops_manual']/1e9:.2f} G")

    comprehensive = {
        "ttft": ttft_results,
        "flops": flops_results,
    }
    comp_out = f"comprehensive_qwen2vl_{ts}.json"
    with open(comp_out, "w") as f:
        json.dump(comprehensive, f, indent=2)
    print(f"\n  Comprehensive results saved → {comp_out}")

    return comprehensive


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2-VL Instruct TTFT + FLOPs benchmark"
    )
    parser.add_argument("--model-path", type=str, default="./Qwen2-VL-2B",
                        help="Qwen2-VL model path (local or HF hub)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="JSON annotation file. 支持: \n"
                             "  GQA:  testdev_balanced_questions.json\n"
                             "  COCO: captions_val2017.json\n"
                             "  省略或 'none': 直接扫描 --image-folder 目录")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="图像目录 (GQA images/ 或 COCO val2017/)")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max samples for TTFT benchmark")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations before timing")
    parser.add_argument("--text-len", type=int, default=30,
                        help="Assumed text token length for FLOPs estimation")
    # Qwen2-VL 原生动态分辨率参数
    # 2B 默认: min=256*28*28=200704, max=1280*28*28=1003520
    # 7B 默认: min=256*28*28=200704, max=16384*28*28=12845056
    parser.add_argument("--min-pixels", type=int, default=None,
                        help="Qwen2-VL min_pixels (None = model default). "
                             "e.g. 256*28*28=200704")
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="Qwen2-VL max_pixels (None = model default). "
                             "e.g. 1280*28*28=1003520 for 2B")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save output JSON files")
    parser.add_argument("--ttft-only", action="store_true",
                        help="Only run TTFT benchmark")
    parser.add_argument("--flops-only", action="store_true",
                        help="Only run FLOPs estimation")
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)

    data_path = args.data_path if args.data_path and args.data_path != "none" else None

    if args.ttft_only:
        run_ttft_benchmark(
            model_path=args.model_path,
            data_path=data_path,
            image_folder=args.image_folder,
            max_samples=args.max_samples,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            warmup=args.warmup,
        )
    elif args.flops_only:
        run_flops_benchmark(
            model_path=args.model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            text_len=args.text_len,
        )
    else:
        args.data_path = data_path  # normalize to None when "none"
        run_comprehensive_test(args)


if __name__ == "__main__":
    main()
