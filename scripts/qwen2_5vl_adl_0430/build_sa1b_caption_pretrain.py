#!/usr/bin/env python3
"""Build a pure-caption SA-1B pretrain JSON for the DAT HD + projector stage.

Difference vs. ``build_sa1b_mix.py``
------------------------------------
- No HR-essential base merge (this dataset is for pretrain, not SFT).
- No AS-Core ``region_caption`` / ``region_vqa`` (those are region-conditional
  VQA, not pure captions; also flagged as noisy in build_sa1b_mix.py).
- No multi-image / Chinese captions by default (keep distribution clean).
- Default ``--max_total 0`` => no cap; consume every IV-en caption whose image
  shard is present on disk. The set grows automatically as more shards are
  downloaded — re-run this script after ``download_sa1b_shards.sh`` to refresh
  the JSON.

Pretrain stage rationale
------------------------
The companion training script
``scripts/qwen2_5vl_adl_0520/exp13_pretrain_sa1b_caption.sh`` freezes the LLM
(no LoRA, ``tune_mm_llm=False``) and the ViT (``tune_mm_vision=False``), and
only unfreezes:

  • ``visual.merger`` (the LLaVA-style projector / PatchMerger)
  • All DAT params (DAT_KEYS_MATCH in llava/train/train_qwen_dat.py:79-86):
    offset prediction (conv_lr_dw, ln_*, conv_lr_proj, proj_intention,
    conv_off_proj), HD KV projection (k_proj_hd, v_proj_hd,
    hd_input_layernorm), and the D3 residual-merge adapter (hd_out_proj,
    hd_gate).

Pure captioning is well suited to this stage:
  - Long single-paragraph descriptions drive the projector to surface every
    visible object, which is the most generic supervisor for image -> text
    alignment.
  - The HD path only contributes when the LR-token attention actually
    benefits from extra detail; caption supervision penalizes
    over-confident details that aren't in the image, so the gate / offset
    network has a stable signal even at large scale.
  - No QA-style answer span means the D1 lr_image injection (Run E1) is the
    only HD entry point — exactly what we want to validate that the
    refactored cross-attention reaches the residual stream.

Usage
-----
    # Default: use every IV-en caption whose shard is on disk
    python scripts/qwen2_5vl_adl_0430/build_sa1b_caption_pretrain.py

    # Cap to 500K (e.g. once 45 shards are downloaded)
    python scripts/qwen2_5vl_adl_0430/build_sa1b_caption_pretrain.py \
        --max_total 500000

Output: /root/autodl-tmp/models_data/sft_data/llava_sa1b_caption_pretrain.json
        (path can be overridden with --output_json)

Also (re)creates the symlink ``train_split/sa1b -> sa1b_images`` so the
training script's ``--image_folder $DATA_ROOT/train_split`` finds the SA-1B
images via the ``sa1b/sa_XXXXXX/sa_Y.jpg`` prefix added by this builder.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sa1b_image_dir",
        type=str,
        default="/root/autodl-tmp/models_data/sa1b_images",
        help="Root dir holding sa_NNNNNN/ subdirs of JPEGs.",
    )
    parser.add_argument(
        "--internvl_dir",
        type=str,
        default="/root/autodl-tmp/models_data/InternVL-SA-1B-Caption",
        help="Dir holding internvl_sa1b_caption_11m_single_image_en.jsonl.",
    )
    parser.add_argument(
        "--internvl_jsonl",
        type=str,
        default="internvl_sa1b_caption_11m_single_image_en.jsonl",
        help="JSONL filename under --internvl_dir. The 11M en single-image "
             "file is the cleanest source; zh / multi-image variants are not "
             "recommended for the pretrain stage.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/root/autodl-tmp/models_data/sft_data/llava_sa1b_caption_pretrain.json",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/models_data/sft_data",
        help="Used to (re)create the symlink train_split/sa1b -> sa1b_image_dir.",
    )
    parser.add_argument(
        "--max_total",
        type=int,
        default=0,
        help="Cap on the number of pretrain samples. 0 (default) = no cap, "
             "use every caption whose image shard is on disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    random.seed(args.seed)

    iv_path = os.path.join(args.internvl_dir, args.internvl_jsonl)
    if not os.path.exists(iv_path):
        raise FileNotFoundError(f"InternVL caption jsonl not found: {iv_path}")

    if not os.path.isdir(args.sa1b_image_dir):
        raise NotADirectoryError(f"sa1b_image_dir not found: {args.sa1b_image_dir}")

    available_shards = sorted(
        d for d in os.listdir(args.sa1b_image_dir) if d.startswith("sa_")
    )
    if not available_shards:
        raise RuntimeError(
            f"No sa_NNNNNN/ subdirs under {args.sa1b_image_dir}. "
            "Did you run download_sa1b_shards.sh?"
        )
    shard_set = set(available_shards)
    print(f"Available shards on disk ({len(available_shards)}): "
          f"{available_shards[0]} ... {available_shards[-1]}")

    print(f"\nScanning IV caption jsonl: {iv_path}")
    raw = []
    with open(iv_path) as f:
        for line in f:
            d = json.loads(line)
            shard = d["image"].split("/")[0]
            if shard in shard_set:
                raw.append(d)
    print(f"  Captions in available shards: {len(raw):,}")

    if args.max_total and len(raw) > args.max_total:
        print(f"  Subsampling to --max_total={args.max_total:,}")
        random.shuffle(raw)
        raw = raw[: args.max_total]

    print(f"\nVerifying image files exist on disk...")
    verified = []
    missing = 0
    for s in raw:
        img_path = os.path.join(args.sa1b_image_dir, s["image"])
        if os.path.exists(img_path):
            verified.append({
                "image": "sa1b/" + s["image"],
                "conversations": s["conversations"],
            })
        else:
            missing += 1
    print(f"  Verified: {len(verified):,}, Missing: {missing}")

    by_shard = Counter(s["image"].split("/")[1] for s in verified)
    print(f"  Verified samples span {len(by_shard)} shards, "
          f"per-shard min/median/max = "
          f"{min(by_shard.values())} / "
          f"{sorted(by_shard.values())[len(by_shard) // 2]} / "
          f"{max(by_shard.values())}")

    random.shuffle(verified)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(verified, f)
    print(f"\nSaved {len(verified):,} samples to: {args.output_json}")

    train_split = os.path.join(args.data_root, "train_split")
    os.makedirs(train_split, exist_ok=True)
    symlink_path = os.path.join(train_split, "sa1b")
    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
        current_target = os.readlink(symlink_path) if os.path.islink(symlink_path) else None
        if current_target == args.sa1b_image_dir:
            print(f"Symlink already correct: {symlink_path} -> {current_target}")
        else:
            print(
                f"WARN: {symlink_path} exists but points to {current_target} "
                f"(want {args.sa1b_image_dir}). Leaving unchanged; remove "
                f"manually if you want this builder to recreate it."
            )
    else:
        os.symlink(args.sa1b_image_dir, symlink_path)
        print(f"Created symlink: {symlink_path} -> {args.sa1b_image_dir}")

    print("\n" + "=" * 60)
    print("Pretrain dataset summary")
    print("=" * 60)
    print(f"  Source jsonl       : {os.path.basename(iv_path)}")
    print(f"  Shards on disk     : {len(available_shards)}")
    print(f"  Samples            : {len(verified):,}")
    print(f"  Avg / shard        : {len(verified) // max(len(by_shard), 1):,}")
    print(f"  Output JSON        : {args.output_json}")
    print(f"  Image folder (sym) : {symlink_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
