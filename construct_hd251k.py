#!/usr/bin/env python3
"""Construct a ~251K high-resolution, task-balanced SFT dataset for DAT training.

Now supports starting from ZERO local files. All annotations and images
are downloaded and converted from Hugging Face by default.

Output mix (SynthDoG capped at 50%):
  synthdog   ~125K  (50.0%)  OCR / document reading   (median 1.1M px)
  docvqa      ~39K  (15.7%)  Document QA              (median 3.8M px)
  chartqa     ~28K  (11.3%)  Chart understanding       (median 446K px)
  infovqa     ~24K  ( 9.5%)  Infographic QA           (median 2.0M px)
  textvqa     ~22K  ( 8.7%)  Scene text               (median 783K px)
  sam          ~9K  ( 3.6%)  General scene (GPT-4V)   (median 3.4M px)
  ai2d         ~3K  ( 1.2%)  Science diagram QA       (median 230K px)
  ──────────────────────────────────────────────────────────────────
  TOTAL      ~251K

All data is downloaded from Hugging Face on first run.
Images are saved to train_split/ subdirectories.
Subsequent runs will reuse cached images.

Usage:
  python construct_hd251k.py                      # from HF only (ZERO local files, recommended)
  python construct_hd251k.py --synthdog_ratio 0.4  # adjust SynthDoG ratio
  python construct_hd251k.py --skip_downloads      # use only cached local data (if available)
  python construct_hd251k.py --from_hf False       # force legacy local JSON mode
  python construct_hd251k.py --force_refresh docvqa,chartqa  # rebuild specific jsonl caches

Resume behavior:
  This script is fully resumable. Every dataset writes a per-dataset jsonl cache
  (e.g. `<SFT_DIR>/cache/docvqa_samples.jsonl`). On re-run, if the cache exists and
  all referenced images are on disk, the dataset is skipped. If a dataset partially
  finished, the jsonl acts as a progress log and processing resumes from the last
  saved index. A failure in one dataset will not abort the whole run.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict

from PIL import Image
from tqdm import tqdm


def _stable_hash(s: str) -> int:
    """Deterministic 32-bit hash, independent of PYTHONHASHSEED.

    Using this instead of built-in `hash()` is critical for resumability:
    we need the same random.sample() indices across separate Python runs.
    """
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "big")

Image.MAX_IMAGE_PIXELS = None

# ===================== Paths =====================
SFT_DIR = "/cluster/data3/wzy/sft_data"
TRAIN_SPLIT = os.path.join(SFT_DIR, "train_split")
CACHE_DIR = os.path.join(SFT_DIR, "cache")          # per-dataset sample jsonl caches
OUTPUT_JSON = os.path.join(SFT_DIR, "llava_hd251k.json")
SEED = 42

# How often to flush the per-dataset jsonl cache while streaming from HF.
CACHE_FLUSH_EVERY = 500

# These are kept for --skip_downloads mode only
HD_SUPPLEMENTS = os.path.join(SFT_DIR, "hd_supplements.jsonl")
BASE_JSON = os.path.join(SFT_DIR, "llava_v1_5_mix665k_shuffled.json")
GPT4V_JSON = os.path.join(SFT_DIR, "sharegpt4v_json",
                          "sharegpt4v_instruct_gpt4-vision_cap100k.json")

FORCE_REFRESH = set()          # filled from CLI in main()
TRUST_EXISTING_IMAGES = set()  # filled from CLI in main()


SYNTHDOG_PROMPTS = [
    "What is written in this image?",
    "Read the text in this document.",
    "Extract all the text from this image.",
    "What does this document say?",
    "Transcribe the text visible in this image.",
]
AI2D_PROMPTS = [
    "Look at the diagram and answer the question.",
    "Based on the diagram, answer the following question.",
    "Use the diagram to answer this question.",
]
# =================================================


def _img_exists(rel_path):
    return os.path.exists(os.path.join(TRAIN_SPLIT, rel_path))


def _hf_download_file(repo_id, filename, local_dir, repo_type="dataset"):
    """Download a single file from HF, preferring modern `hf` CLI."""
    os.makedirs(local_dir, exist_ok=True)
    target_path = os.path.join(local_dir, filename)
    if os.path.exists(target_path):
        return target_path

    cmds = []
    if shutil.which("hf"):
        cmds.append([
            "hf", "download", repo_id, filename,
            "--repo-type", repo_type,
            "--local-dir", local_dir,
        ])
    if shutil.which("huggingface-cli"):
        cmds.append([
            "huggingface-cli", "download", repo_id, filename,
            "--repo-type", repo_type,
            "--local-dir", local_dir,
        ])

    last_err = None
    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception as e:
            last_err = RuntimeError(f"failed to run {' '.join(cmd)}: {e}")
            continue

        if proc.returncode == 0 and os.path.exists(target_path):
            return target_path

        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr if stderr else stdout
        if proc.returncode != 0:
            last_err = RuntimeError(
                f"{' '.join(cmd)} failed with code {proc.returncode}. {detail}"
            )
        else:
            last_err = FileNotFoundError(
                f"{' '.join(cmd)} succeeded but `{target_path}` not found."
            )

    # Final fallback: Python SDK (works even when CLI behavior changes)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=local_dir,
        )
        if os.path.exists(target_path):
            return target_path
        if os.path.exists(downloaded):
            shutil.copy2(downloaded, target_path)
            return target_path
    except Exception as e:
        sdk_err = e
    else:
        sdk_err = None

    raise RuntimeError(
        f"Failed to download `{filename}` from `{repo_id}`. "
        f"Last CLI error: {last_err}. SDK error: {sdk_err}"
    )


def _print_counts(samples, label=""):
    counts = defaultdict(int)
    for s in samples:
        counts[s.get("image", "").split("/")[0]] += 1
    if label:
        print(f"  {label}:")
    for k in sorted(counts):
        print(f"    {k}: {counts[k]}")
    print(f"    total: {len(samples)}")


# ────────────────────────────────────────────────
#  Resume helpers (per-dataset jsonl cache)
# ────────────────────────────────────────────────

def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}_samples.jsonl")


def _meta_path(name):
    """Sidecar JSON that remembers how many samples the source actually produced
    the last time this dataset was built end-to-end. Lets us distinguish
    "source smaller than requested target" from "interrupted mid-run"."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}_meta.json")


def _read_meta(name):
    path = _meta_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_meta(name, meta):
    path = _meta_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_cached_samples(name, require_images=True):
    """Load previously cached samples for `name`. Drops entries whose images are missing."""
    path = _cache_path(name)
    if name in FORCE_REFRESH or not os.path.exists(path):
        return [], set()
    samples = []
    seen_ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if require_images and not _img_exists(item.get("image", "")):
                continue
            if item.get("id") in seen_ids:
                continue
            seen_ids.add(item["id"])
            samples.append(item)
    return samples, seen_ids


def _append_samples_to_cache(name, samples):
    """Append new samples to the per-dataset jsonl cache (flush immediately)."""
    if not samples:
        return
    path = _cache_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _rewrite_cache(name, samples):
    """Overwrite the per-dataset jsonl cache with `samples` (used after dedup)."""
    path = _cache_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _wipe_dataset(name, wipe_images=True):
    """Remove per-dataset jsonl cache and (optionally) its train_split/<name>/ folder.

    Used when a dataset's resume state is incomplete and we have to restart.
    The image folder is wiped because the mapping (local-filename -> ds-row)
    depends on a seed that can change between runs, so stale images from a
    previous run could become mis-aligned with newly generated annotations.
    """
    cache = _cache_path(name)
    if os.path.exists(cache):
        os.remove(cache)
        print(f"    wiped cache: {cache}")
    meta = _meta_path(name)
    if os.path.exists(meta):
        os.remove(meta)
    if wipe_images:
        img_dir = os.path.join(TRAIN_SPLIT, name)
        if os.path.isdir(img_dir):
            n = len(os.listdir(img_dir))
            shutil.rmtree(img_dir)
            print(f"    wiped {n} stale images: {img_dir}")


# ────────────────────────────────────────────────
#  Step 1: High-res data from Hugging Face (docvqa / infovqa / chartqa / textvqa)
# ────────────────────────────────────────────────

def _try_load_dataset(dataset_name, subset_name, split):
    """Thin wrapper around datasets.load_dataset without deprecated trust_remote_code."""
    from datasets import load_dataset
    return load_dataset(dataset_name, subset_name, split=split)


def process_vqa_from_hf(dataset_name, split, subset_name, save_dir_name,
                        target_count=None, question_key="question",
                        answer_key="answers", default_question="What is written in this image?",
                        fallbacks=None):
    """Download VQA dataset from HF, save images, convert to LLaVA format.

    Dataset-level resume: if the on-disk cache jsonl already contains
    `target_count` valid samples (all images present), we reuse it. Otherwise
    we wipe any partial state (images + cache) and re-download from scratch.
    This avoids hard-to-debug misalignments between cached images and newly
    sampled annotations, at the cost of redoing partially finished runs.

    `fallbacks` is an optional list of (dataset_name, subset_name, split,
    question_key, answer_key) tuples to try if the primary source fails.
    """
    trust_existing = save_dir_name in TRUST_EXISTING_IMAGES

    # ── 1. If forced refresh, wipe up front. Otherwise only keep fully-complete caches. ──
    if save_dir_name in FORCE_REFRESH:
        print(f"  [{save_dir_name}] --force_refresh requested, wiping previous state.")
        _wipe_dataset(save_dir_name, wipe_images=True)

    # When trusting existing images, `_img_exists` shouldn't gate cache reuse:
    # a cached entry is valid as long as its referenced image file exists on disk.
    cached, _ = _load_cached_samples(save_dir_name, require_images=True)
    meta = _read_meta(save_dir_name)

    # Figure out the *real* completion target. The HF source may have fewer
    # rows than `target_count` (e.g. Ahren09/InfoVQA only ships ~23.9k train
    # rows, so asking for 24000 will forever look "incomplete"). If a previous
    # end-to-end run already drained the source under an equal-or-larger ask,
    # accept its `final_count` as the actual ceiling.
    effective_target = target_count
    source_capped = False
    if (
        meta
        and meta.get("source_exhausted")
        and isinstance(meta.get("final_count"), int)
        and isinstance(meta.get("target_count"), int)
        and target_count is not None
        and meta["target_count"] >= target_count
    ):
        effective_target = min(target_count, meta["final_count"])
        source_capped = effective_target < target_count

    complete_threshold = effective_target if effective_target else max(len(cached), 1)
    if cached and len(cached) >= complete_threshold:
        if source_capped:
            print(f"  [{save_dir_name}] cache is complete: {len(cached)} samples "
                  f"(source caps at {meta['final_count']} < requested {target_count}). "
                  f"Skipping HF download.")
        else:
            print(f"  [{save_dir_name}] cache is complete: {len(cached)} samples "
                  f"(target {target_count}). Skipping HF download.")
        return cached[:target_count] if target_count else cached

    if trust_existing:
        # Keep whatever is on disk; just (re)build the jsonl annotations.
        # WARNING: because the old image filenames may have come from a
        # different hash seed than we now use, the QA at index `i` may
        # point to image file `<name>_{i:06d}.jpg` that physically shows
        # a DIFFERENT document than what the QA is asking about.
        img_dir = os.path.join(TRAIN_SPLIT, save_dir_name)
        on_disk = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
        print(f"  [{save_dir_name}] --trust_existing_images ON: keeping {on_disk} "
              f"images + wiping only the (stale) jsonl cache.")
        _wipe_dataset(save_dir_name, wipe_images=False)
    elif cached or os.path.isdir(os.path.join(TRAIN_SPLIT, save_dir_name)):
        have = len(cached)
        img_dir = os.path.join(TRAIN_SPLIT, save_dir_name)
        on_disk = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
        print(f"  [{save_dir_name}] incomplete state: {have} cached samples, "
              f"{on_disk} images on disk, target {target_count} "
              f"(effective {effective_target}). Restarting dataset.")
        _wipe_dataset(save_dir_name, wipe_images=True)

    # ── 2. Load the HF dataset (primary + fallbacks) ──
    sources = [(dataset_name, subset_name, split, question_key, answer_key)]
    if fallbacks:
        sources.extend(fallbacks)

    ds = None
    used_source = None
    last_err = None
    for src in sources:
        d_name, s_name, s_split, _q_key, _a_key = src
        try:
            print(f"  Loading {d_name} ({s_name or 'default'}, split={s_split}) -> {save_dir_name} ...")
            ds = _try_load_dataset(d_name, s_name, s_split)
            used_source = src
            break
        except Exception as e:
            last_err = e
            print(f"    ! Failed to load {d_name}: {type(e).__name__}: {e}")

    if ds is None:
        print(f"  [{save_dir_name}] ERROR: all sources failed ({last_err}). Returning [].")
        return []

    _, _, _, question_key, answer_key = used_source

    # ── 3. Reservoir / subsample with a *stable* seed ──
    if target_count and len(ds) > target_count:
        indices = random.Random(SEED + _stable_hash(save_dir_name)).sample(range(len(ds)), target_count)
        ds = ds.select(indices)

    save_dir = os.path.join(TRAIN_SPLIT, save_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── 4. Process with incremental jsonl writes (for progress visibility only;
    #       if this run is interrupted it will be wiped on next run). ──
    samples = []
    buffer = []

    def _flush():
        nonlocal buffer
        if buffer:
            _append_samples_to_cache(save_dir_name, buffer)
            buffer = []

    try:
        for idx, item in enumerate(tqdm(ds, desc=f"    {save_dir_name}")):
            sample_id = f"{save_dir_name}_{idx}"

            img = item.get("image")
            fname = f"{save_dir_name}_{idx:06d}.jpg"
            img_path = os.path.join(save_dir, fname)

            if trust_existing and os.path.exists(img_path):
                pass
            else:
                if img is None:
                    continue
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                try:
                    img.save(img_path)
                except Exception as e:
                    tqdm.write(f"    ! save image {fname} failed: {e}")
                    continue

            question = item.get(question_key, default_question)
            if isinstance(question, list):
                question = question[0] if question else default_question

            answers = item.get(answer_key, [""])
            if isinstance(answers, list) and answers:
                answer = answers[0] if isinstance(answers[0], str) else str(answers[0])
            else:
                answer = str(answers) if answers else "N/A"

            sample = {
                "id": sample_id,
                "image": f"{save_dir_name}/{fname}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            }
            samples.append(sample)
            buffer.append(sample)

            if len(buffer) >= CACHE_FLUSH_EVERY:
                _flush()
    finally:
        _flush()

    # Record how many samples the source actually produced. If it's below
    # `target_count`, future runs treat `final_count` as the real ceiling
    # and skip re-downloading (see the meta check at the top of this func).
    if target_count is not None:
        _write_meta(save_dir_name, {
            "final_count": len(samples),
            "target_count": target_count,
            "source_exhausted": len(samples) < target_count,
        })

    print(f"    {save_dir_name}: {len(samples)} samples (cache -> {_cache_path(save_dir_name)})")
    return samples


def _safe_collect(fn, label):
    """Run `fn()` and wrap any failure so one bad dataset doesn't kill the whole run."""
    try:
        return fn()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  [{label}] ERROR: {type(e).__name__}: {e}. Skipping this dataset.")
        import traceback
        traceback.print_exc()
        return []


def collect_hd_from_hf():
    """Download and convert all HD VQA datasets from Hugging Face (no local files needed)."""
    samples = []

    samples.extend(_safe_collect(lambda: process_vqa_from_hf(
        "HuggingFaceM4/DocumentVQA", "train", None, "docvqa",
        target_count=39000, question_key="question", answer_key="answers",
    ), "docvqa"))

    samples.extend(_safe_collect(lambda: process_vqa_from_hf(
        "HuggingFaceM4/ChartQA", "train", None, "chartqa",
        target_count=28000, question_key="query", answer_key="label",
        default_question="What does this chart show?",
    ), "chartqa"))

    # InfoVQA: the old `lmms-lab/infographicvqa` has been removed from HF.
    # Use `Ahren09/InfoVQA` (train 23.9k / val 2.8k / test 3.29k, same schema),
    # and fall back to `vidore/infovqa_train` (10.1k, different schema) if needed.
    samples.extend(_safe_collect(lambda: process_vqa_from_hf(
        "Ahren09/InfoVQA", "train", None, "infovqa",
        target_count=24000, question_key="question", answer_key="answer",
        default_question="What is shown in this infographic?",
        fallbacks=[
            ("vidore/infovqa_train", None, "train", "query", "answer"),
        ],
    ), "infovqa"))

    samples.extend(_safe_collect(lambda: process_vqa_from_hf(
        "textvqa", "train", None, "textvqa",
        target_count=22000, question_key="question", answer_key="answers",
        default_question="What is written in this image?",
        fallbacks=[
            ("lmms-lab/textvqa", None, "train", "question", "answers"),
            ("facebook/textvqa", None, "train", "question", "answers"),
        ],
    ), "textvqa"))

    _print_counts(samples, "HD from HF")
    return samples


# ────────────────────────────────────────────────
#  Step 2: SynthDoG (fix existing + download new)
# ────────────────────────────────────────────────

def _parse_synthdog_gt(gt_str):
    try:
        return json.loads(gt_str).get("gt_parse", {}).get("text_sequence", "").strip()
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""


def collect_synthdog(target_count):
    """Build SynthDoG samples with correct ground-truth annotations.

    Resumable: cached samples are kept in `<CACHE_DIR>/synthdog_samples.jsonl` and
    flushed every CACHE_FLUSH_EVERY new entries.
    """
    from datasets import load_dataset

    existing_count = 200_000
    cached, seen_ids = _load_cached_samples("synthdog", require_images=True)
    if cached and len(cached) >= target_count:
        print(f"  [synthdog] reusing {len(cached)} cached samples (target {target_count}).")
        return cached[:target_count]

    print(f"  Loading SynthDoG HF dataset (target {target_count}, have {len(cached)}) ...")
    try:
        ds = load_dataset("naver-clova-ix/synthdog-en", split="train")
    except Exception as e:
        print(f"  WARNING: Could not load SynthDoG ({e}). Returning {len(cached)} cached.")
        return cached

    rng = random.Random(SEED)
    existing_hf_indices = rng.sample(range(len(ds)), existing_count)

    samples = list(cached)
    prompt_rng = random.Random(SEED + 100)
    buffer = []

    def _flush():
        nonlocal buffer
        if buffer:
            _append_samples_to_cache("synthdog", buffer)
            buffer = []

    try:
        print(f"  Fixing annotations for {existing_count} existing images ...")
        for loop_idx, hf_idx in enumerate(tqdm(existing_hf_indices, desc="    SynthDoG fix")):
            if len(samples) >= target_count:
                break
            sample_id = f"synthdog_{loop_idx}"
            if sample_id in seen_ids:
                continue
            fname = f"synthdog_{loop_idx}.jpg"
            if not _img_exists(f"synthdog/{fname}"):
                continue
            text = _parse_synthdog_gt(ds[hf_idx].get("ground_truth", ""))
            if not text:
                continue
            sample = {
                "id": sample_id,
                "image": f"synthdog/{fname}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{prompt_rng.choice(SYNTHDOG_PROMPTS)}"},
                    {"from": "gpt", "value": text},
                ],
            }
            samples.append(sample)
            seen_ids.add(sample_id)
            buffer.append(sample)
            if len(buffer) >= CACHE_FLUSH_EVERY:
                _flush()

        if len(samples) < target_count:
            shortfall = target_count - len(samples)
            print(f"  Downloading {shortfall} new SynthDoG images ...")
            used = set(existing_hf_indices)
            available = [i for i in range(len(ds)) if i not in used]
            random.Random(SEED + 2).shuffle(available)
            save_dir = os.path.join(TRAIN_SPLIT, "synthdog")
            os.makedirs(save_dir, exist_ok=True)
            offset = existing_count

            for loop_idx, hf_idx in enumerate(tqdm(available[:shortfall], desc="    SynthDoG new")):
                sample_id = f"synthdog_{offset + loop_idx}"
                if sample_id in seen_ids:
                    continue
                item = ds[hf_idx]
                text = _parse_synthdog_gt(item.get("ground_truth", ""))
                if not text:
                    continue
                fname = f"synthdog_{offset + loop_idx}.jpg"
                path = os.path.join(save_dir, fname)
                if not os.path.exists(path):
                    try:
                        item["image"].convert("RGB").save(path)
                    except Exception as e:
                        tqdm.write(f"    ! save synthdog {fname} failed: {e}")
                        continue
                sample = {
                    "id": sample_id,
                    "image": f"synthdog/{fname}",
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt_rng.choice(SYNTHDOG_PROMPTS)}"},
                        {"from": "gpt", "value": text},
                    ],
                }
                samples.append(sample)
                seen_ids.add(sample_id)
                buffer.append(sample)
                if len(buffer) >= CACHE_FLUSH_EVERY:
                    _flush()
    finally:
        _flush()

    print(f"    SynthDoG total: {len(samples)} (cache -> {_cache_path('synthdog')})")
    return samples


# ────────────────────────────────────────────────
#  Step 3: SAM general scene VQA (ShareGPT4V)
# ────────────────────────────────────────────────

def collect_sam():
    """Match local SAM images to GPT4V 100K annotations."""
    cached, _ = _load_cached_samples("sam", require_images=True)
    if cached:
        print(f"  [sam] reusing {len(cached)} cached samples.")
        return cached

    sam_dir = os.path.join(TRAIN_SPLIT, "sam", "images")

    if not os.path.isdir(sam_dir) or not os.listdir(sam_dir):
        print("  SAM images not found, downloading from HuggingFace ...")
        _download_sam_images(sam_dir)

    local_imgs = {f for f in os.listdir(sam_dir) if f.endswith(".jpg")}
    print(f"  Local SAM images: {len(local_imgs)}")

    if not os.path.exists(GPT4V_JSON):
        print("  GPT4V annotation JSON not found, downloading ...")
        _hf_download_file(
            "Lin-Chen/ShareGPT4V",
            "sharegpt4v_instruct_gpt4-vision_cap100k.json",
            os.path.dirname(GPT4V_JSON),
            repo_type="dataset",
        )

    with open(GPT4V_JSON) as f:
        gpt4v = json.load(f)

    samples = []
    for item in gpt4v:
        img = item.get("image", "")
        if "sam/" in img and img.split("/")[-1] in local_imgs:
            samples.append(item)

    print(f"    sam (GPT4V matched): {len(samples)}")
    _rewrite_cache("sam", samples)
    return samples


def _download_sam_images(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    zip_dir = os.path.join(SFT_DIR, "sharegpt4v_sam_dl")
    zip_path = os.path.join(zip_dir, "sam_images_share-sft.zip")

    if not os.path.exists(zip_path):
        _hf_download_file(
            "DavidNguyen/ShareGPT4V-Sam",
            "sam_images_share-sft.zip",
            zip_dir,
            repo_type="dataset",
        )

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"SAM zip not found after download: {zip_path}")

    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in tqdm(zf.namelist(), desc="    Unzip SAM"):
            if member.endswith(".jpg"):
                data = zf.read(member)
                out = os.path.join(target_dir, os.path.basename(member))
                with open(out, "wb") as fout:
                    fout.write(data)


# ────────────────────────────────────────────────
#  Step 4: AI2D science diagrams
# ────────────────────────────────────────────────

def collect_ai2d():
    """Download AI2D from HuggingFace and convert to LLaVA format (resumable)."""
    cached, seen_ids = _load_cached_samples("ai2d", require_images=True)
    if cached and len(cached) >= 3000:
        print(f"  [ai2d] reusing {len(cached)} cached samples.")
        return cached

    save_dir = os.path.join(TRAIN_SPLIT, "ai2d")
    os.makedirs(save_dir, exist_ok=True)

    if not cached and os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 3000:
        print("  AI2D images already on disk, loading annotations ...")
        legacy = _load_existing_ai2d(save_dir)
        if legacy:
            _rewrite_cache("ai2d", legacy)
            return legacy

    print("  Downloading AI2D from lmms-lab/ai2d ...")
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/ai2d", split="test", streaming=True)

    samples = list(cached)
    rng = random.Random(SEED + 50)
    buffer = []

    def _flush():
        nonlocal buffer
        if buffer:
            _append_samples_to_cache("ai2d", buffer)
            buffer = []

    try:
        for idx, item in enumerate(tqdm(ds, desc="    AI2D")):
            sample_id = f"ai2d_{idx}"
            if sample_id in seen_ids:
                continue
            fname = f"ai2d_{idx}.png"
            path = os.path.join(save_dir, fname)
            if not os.path.exists(path):
                try:
                    item["image"].save(path)
                except Exception as e:
                    tqdm.write(f"    ! save ai2d {fname} failed: {e}")
                    continue

            question = item.get("question", "")
            options = item.get("options", [])
            answer_idx = item.get("answer", 0)

            if options:
                try:
                    answer_val = options[int(answer_idx)]
                except (IndexError, ValueError, TypeError):
                    answer_val = str(answer_idx)
                choice_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(options))
                q_text = f"{rng.choice(AI2D_PROMPTS)}\n{question}\n{choice_str}"
            else:
                q_text = question
                answer_val = str(answer_idx)

            sample = {
                "id": sample_id,
                "image": f"ai2d/{fname}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{q_text}"},
                    {"from": "gpt", "value": answer_val},
                ],
            }
            samples.append(sample)
            seen_ids.add(sample_id)
            buffer.append(sample)
            if len(buffer) >= CACHE_FLUSH_EVERY:
                _flush()
    finally:
        _flush()

    print(f"    ai2d: {len(samples)} (cache -> {_cache_path('ai2d')})")
    return samples


def _load_existing_ai2d(save_dir):
    """Re-create AI2D annotations from on-disk images (no HF needed)."""
    jsonl = os.path.join(SFT_DIR, "ai2d_samples.jsonl")
    if os.path.exists(jsonl):
        samples = []
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if _img_exists(item["image"]):
                        samples.append(item)
        print(f"    ai2d (from cache): {len(samples)}")
        return samples
    return []


# ────────────────────────────────────────────────
#  Main assembly
# ────────────────────────────────────────────────

def _print_resume_state():
    """Quick summary of what is already cached, so the user knows resume status."""
    print("\n  Resume state (per-dataset cache under "
          f"{CACHE_DIR}):")
    if not os.path.isdir(CACHE_DIR):
        print("    (no cache yet — this will be a fresh run)")
        return
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith("_samples.jsonl"):
            continue
        path = os.path.join(CACHE_DIR, fname)
        name = fname.replace("_samples.jsonl", "")
        try:
            with open(path) as f:
                n = sum(1 for line in f if line.strip())
        except OSError:
            n = 0
        marker = "  (will be refreshed)" if name in FORCE_REFRESH else ""
        print(f"    {name:<12} {n:>7} cached samples{marker}")


def main():
    parser = argparse.ArgumentParser(description="Construct HD-251K SFT dataset")
    parser.add_argument("--synthdog_ratio", type=float, default=0.5,
                        help="SynthDoG proportion in final dataset (default 0.5)")
    parser.add_argument("--output", type=str, default=OUTPUT_JSON)
    parser.add_argument("--skip_downloads", action="store_true",
                        help="Skip HuggingFace downloads, use only existing local data")
    parser.add_argument("--from_hf", action="store_true", default=True,
                        help="Download all annotations from Hugging Face (default: True, zero local files)")
    parser.add_argument("--force_refresh", type=str, default="",
                        help="Comma-separated dataset names to force re-download "
                             "(e.g. 'infovqa,textvqa'). Ignores existing cache.")
    parser.add_argument("--trust_existing_images", type=str, default="",
                        help="Comma-separated dataset names for which existing images "
                             "on disk should be reused (only the jsonl cache is rebuilt). "
                             "WARNING: image-QA alignment is NOT guaranteed if the old "
                             "images came from a previous run with a different hash seed.")
    args = parser.parse_args()

    random.seed(SEED)
    global FORCE_REFRESH, TRUST_EXISTING_IMAGES
    FORCE_REFRESH = {n.strip() for n in args.force_refresh.split(",") if n.strip()}
    TRUST_EXISTING_IMAGES = {n.strip() for n in args.trust_existing_images.split(",") if n.strip()}

    _print_resume_state()
    print("\n  Resume policy:")
    print("    - docvqa/chartqa/infovqa/textvqa : dataset-level (cache >= target -> keep;")
    print("      otherwise wipe images + cache and re-download the whole dataset)")
    print("    - synthdog / ai2d / sam          : sample-level (incremental resume)")
    print("    Use --force_refresh=<name1,name2,...> to force redownload a dataset.")
    if TRUST_EXISTING_IMAGES:
        print()
        print("  " + "!" * 68)
        print(f"  !!  --trust_existing_images={sorted(TRUST_EXISTING_IMAGES)}")
        print("  !!  Existing images will be reused; only jsonl annotations will be rebuilt.")
        print("  !!  Image-QA alignment is NOT guaranteed for these datasets.")
        print("  !!  Downstream training quality MAY be degraded.")
        print("  " + "!" * 68)

    print("=" * 60)
    if args.from_hf and not args.skip_downloads:
        print("Step 1: High-res data from Hugging Face (docvqa, infovqa, chartqa, textvqa)")
        print("=" * 60)
        non_synthdog = collect_hd_from_hf()
    else:
        print("Step 1: Local high-res data (docvqa, infovqa, chartqa, textvqa)")
        print("=" * 60)
        print("  WARNING: --from_hf=False requires pre-existing hd_supplements.jsonl and base JSON.")
        print("  Falling back to HF mode for now.")
        non_synthdog = collect_hd_from_hf()

    if not args.skip_downloads:
        print(f"\n{'=' * 60}")
        print("Step 3: SAM general scene VQA")
        print("=" * 60)
        non_synthdog.extend(_safe_collect(collect_sam, "sam"))

        print(f"\n{'=' * 60}")
        print("Step 4: AI2D science diagrams")
        print("=" * 60)
        non_synthdog.extend(_safe_collect(collect_ai2d, "ai2d"))
    else:
        # Try loading from existing cache files
        sam_jsonl = os.path.join(SFT_DIR, "sharegpt4v_sam_matched.jsonl")
        if os.path.exists(sam_jsonl):
            with open(sam_jsonl) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if _img_exists(item["image"]):
                            non_synthdog.append(item)
        ai2d_jsonl = os.path.join(SFT_DIR, "ai2d_samples.jsonl")
        if os.path.exists(ai2d_jsonl):
            with open(ai2d_jsonl) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if _img_exists(item["image"]):
                            non_synthdog.append(item)

    non_sd_count = len(non_synthdog)
    synthdog_target = int(non_sd_count * args.synthdog_ratio / (1 - args.synthdog_ratio))
    total_target = non_sd_count + synthdog_target
    print(f"\n  Non-SynthDoG: {non_sd_count}")
    print(f"  SynthDoG target ({args.synthdog_ratio:.0%}): {synthdog_target}")
    print(f"  Total target: {total_target}")

    # ── Collect SynthDoG ──
    print(f"\n{'=' * 60}")
    print(f"Step 2: SynthDoG ({synthdog_target} samples)")
    print("=" * 60)
    if args.skip_downloads:
        synthdog_jsonl = os.path.join(SFT_DIR, "synthdog_fixed.jsonl")
        if os.path.exists(synthdog_jsonl):
            pool = []
            with open(synthdog_jsonl) as f:
                for line in f:
                    if line.strip():
                        pool.append(json.loads(line))
        else:
            print("  WARNING: No cached SynthDoG data and --skip_downloads set.")
            pool = []
        rng = random.Random(SEED)
        synthdog_samples = rng.sample(pool, min(synthdog_target, len(pool))) if pool else []
    else:
        synthdog_samples = _safe_collect(lambda: collect_synthdog(synthdog_target), "synthdog")

    # ── Merge, shuffle, write ──
    print(f"\n{'=' * 60}")
    print("Final: Merge, shuffle & write")
    print("=" * 60)

    final = non_synthdog + synthdog_samples
    random.Random(SEED + 999).shuffle(final)

    final_counts = defaultdict(int)
    for item in final:
        final_counts[item.get("image", "").split("/")[0]] += 1

    print("\n  Dataset breakdown:")
    for k in sorted(final_counts):
        cnt = final_counts[k]
        pct = cnt * 100 / len(final)
        print(f"    {k:<14} {cnt:>8}  ({pct:>5.1f}%)")
    print(f"    {'TOTAL':<14} {len(final):>8}")

    print(f"\n  Writing {args.output} ...")
    with open(args.output, "w") as f:
        json.dump(final, f)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n  Done! {len(final)} samples, {size_mb:.1f} MB")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
