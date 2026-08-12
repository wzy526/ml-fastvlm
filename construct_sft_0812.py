#!/usr/bin/env python3
"""Build the 0812 SFT mix: base ivcap 369k + OCR train sets + visual-search data.

Composition (see chat 0812 / advisor: DeepEyes, Mini-o3, vision-OPD line):

  base       369k  llava_hr_essential_sa1b_ivcap.json  (unchanged)
  chartqa     28k  from llava_hd251k.json   (ChartQA train, already on disk)
  textvqa     22k  from llava_hd251k.json   (TextVQA train, already on disk)
  ai2d         3k  from llava_hd251k.json   (lmms-lab/ai2d TEST split — drop it
                                             with --no_ai2d if ai2d ever joins
                                             the eval suite)
  stvqa      ~26k  NEW download  (scene text, complements TextVQA)
  deepeyes   ~25k  NEW download  (DeepEyes-47k visual_toolbox subsample:
                                  high-res visual search w/ verifiable
                                  answers, Apache 2.0; thinklite math
                                  subset intentionally excluded)
  visualprobe ~5k  NEW download  (Mini-o3 VisualProbe train: hardest visual
                                  search, CC BY-NC research only)

The three NEW sets double as the RL / on-policy-distillation prompt pool
later (verifiable short answers -> exact-match reward).

RUN THIS ON THE TRAINING CLUSTER (or any box with >8GB RAM + HF mirror):
the DeepEyes parquet shards are single ~1-2GB row groups, so pyarrow must
decompress a whole row group at once — that alone exceeds the 2GB cgroup cap
on the autodl code box. The merge phase also holds the full ~500k-sample mix
in memory (~3GB).

Usage:
  python construct_sft_0812.py                  # download + merge
  python construct_sft_0812.py --skip_downloads # merge only
  python construct_sft_0812.py --download_only  # downloads only, no merge
  python construct_sft_0812.py --only stvqa,visualprobe   # subset of downloads
  python construct_sft_0812.py --inspect        # print converted samples of
                                                #  the new sets for eyeballing
  python construct_sft_0812.py --no_ai2d        # exclude ai2d (test split)

Env:
  SFT_DIR   sft_data root (default /root/autodl-tmp/models_data/sft_data;
            on the training cluster:
            /data/oss_bucket_0/wangziyi/models_data/sft_data)
  HF_HOME / HF_HUB_CACHE   point at a large local disk

Images land in $SFT_DIR/train_split/{stvqa,deepeyes,visualprobe}/ and
per-dataset LLaVA-format fragments in $SFT_DIR/extra_0812/<name>.json, so the
download phase is resumable and the merge phase is fully offline.

After building on the training cluster, add the new prefixes to the local
symlink farm before SFT (OSS FUSE can't hold symlinks):
  OSS_TS=/data/oss_bucket_0/wangziyi/models_data/sft_data/train_split
  for d in stvqa deepeyes visualprobe chartqa textvqa ai2d; do
      ln -sfn "$OSS_TS/$d" /home/pingping.wzy/sft_data/train_split/$d
  done
"""

import argparse
import io
import json
import os
import random

SFT_DIR = os.environ.get("SFT_DIR", "/root/autodl-tmp/models_data/sft_data")
TRAIN_SPLIT = os.path.join(SFT_DIR, "train_split")
EXTRA_DIR = os.path.join(SFT_DIR, "extra_0812")

BASE_JSON = os.path.join(SFT_DIR, "llava_hr_essential_sa1b_ivcap.json")
HD251K_JSON = os.path.join(SFT_DIR, "llava_hd251k.json")
OUTPUT_JSON = os.path.join(SFT_DIR, "llava_hr_ocr_vs_0812.json")

HD251K_TAKE_PREFIXES = ["chartqa", "textvqa", "ai2d"]

SEED = 42

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# DeepEyes-47k shards to use: the two visual-search ("visual_toolbox") shards.
# data_thinklite_reasoning_acc.parquet is a math-reasoning subset — excluded.
DEEPEYES_REPO = "ChenShawn/DeepEyes-Datasets-47k"
DEEPEYES_SHARDS = [
    "data_0.1.2_visual_toolbox_v2.parquet",
    "data_v0.8_visual_toolbox_v2.parquet",
]

VISUALPROBE_REPO = "Mini-o3/VisualProbe_train"
STVQA_REPO = "vikhyatk/st-vqa"


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def _fragment_path(name):
    return os.path.join(EXTRA_DIR, f"{name}.json")


def _load_fragment(name):
    p = _fragment_path(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _save_fragment(name, samples):
    os.makedirs(EXTRA_DIR, exist_ok=True)
    with open(_fragment_path(name), "w") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"  [{name}] wrote {len(samples)} samples -> {_fragment_path(name)}")


def _save_jpg(img, save_dir, fname):
    path = os.path.join(save_dir, fname)
    if os.path.exists(path):
        return True
    try:
        img.convert("RGB").save(path, quality=92)
        return True
    except Exception as e:
        print(f"    ! save {fname} failed: {e}")
        return False


def _make_sample(prefix, sample_id, fname, question, answer):
    return {
        "id": f"{prefix}_{sample_id}",
        "image": f"{prefix}/{fname}",
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


# RL prompts in DeepEyes/VisualProbe carry agentic tool-use instructions
# ("Think first, call **image_zoom_in_tool** ... <think>...</think>"). Our
# model has no tools — keeping these would teach a broken output format, so
# cut the question at the first marker occurrence.
_RL_INSTRUCTION_MARKERS = (
    "Think first, call",
    "Format strictly as",
    "image_zoom_in_tool",
    "<think>",
)


def _clean_question(q):
    """Strip image placeholders, chat-template remnants and RL tool suffixes."""
    if not isinstance(q, str):
        return None
    for tok in ("<image>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
        q = q.replace(tok, "")
    for marker in _RL_INSTRUCTION_MARKERS:
        idx = q.find(marker)
        if idx != -1:
            q = q[:idx]
    q = q.strip()
    return q or None


def _valid_short_answer(answer):
    """Keep short, verifiable text answers; drop bbox/JSON ground truths."""
    if not isinstance(answer, str):
        return None
    answer = answer.strip()
    if not answer or len(answer) > 100 or answer[0] in "[{":
        return None
    return answer


def _show(name, question, answer):
    print(f"\n  [{name} sample] Q: {question[:300]}\n{' ' * (len(name) + 12)}A: {answer}")


# ---------------------------------------------------------------------------
#  ST-VQA  (vikhyatk/st-vqa: rows of {image, qas=[{question, answers}]})
# ---------------------------------------------------------------------------

def collect_stvqa(target_count, inspect=False):
    name = "stvqa"
    cached = _load_fragment(name)
    if cached is not None:
        print(f"  [{name}] fragment exists ({len(cached)} samples). Skipping download.")
        return cached

    from datasets import load_dataset
    from tqdm import tqdm

    ds = load_dataset(STVQA_REPO, split="train")
    print(f"  [{name}] loaded {STVQA_REPO} ({len(ds)} images)")

    save_dir = os.path.join(TRAIN_SPLIT, name)
    os.makedirs(save_dir, exist_ok=True)

    samples, shown = [], 0
    for idx, item in enumerate(tqdm(ds, desc=f"    {name}")):
        if target_count and len(samples) >= target_count:
            break
        img = item.get("image")
        if img is None or not hasattr(img, "convert"):
            continue
        qas = item.get("qas") or []
        fname = f"{name}_{idx:06d}.jpg"
        saved = False
        for qi, qa in enumerate(qas):
            if target_count and len(samples) >= target_count:
                break
            question = _clean_question(qa.get("question"))
            answers = qa.get("answers")
            answer = answers[0] if isinstance(answers, list) and answers else answers
            answer = _valid_short_answer(answer)
            if not question or not answer:
                continue
            if not saved:
                if not _save_jpg(img, save_dir, fname):
                    break
                saved = True
            samples.append(_make_sample(name, f"{idx}_{qi}", fname, question, answer))
            if inspect and shown < 5:
                _show(name, question, answer)
                shown += 1

    print(f"  [{name}] kept {len(samples)} QA pairs")
    _save_fragment(name, samples)
    return samples


# ---------------------------------------------------------------------------
#  DeepEyes-47k  (VeRL parquet shards, streamed with pyarrow in small batches)
# ---------------------------------------------------------------------------

def _extract_qa_verl(row):
    """(question, answer) from a VeRL-style row dict, or None to skip."""
    question = None
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                question = msg.get("content")
    elif isinstance(prompt, str):
        question = prompt
    if question is None:
        question = row.get("problem", row.get("question"))
    question = _clean_question(question)
    if not question:
        return None

    answer = None
    rm = row.get("reward_model")
    if isinstance(rm, dict):
        answer = rm.get("ground_truth")
    if answer is None:
        answer = row.get("solution", row.get("answer", row.get("ground_truth")))
    if isinstance(answer, list):
        answer = answer[0] if answer else None
    answer = _valid_short_answer(answer)
    if not answer:
        return None
    return question, answer


def _row_single_image(row):
    """PIL image if the row has exactly one image, else None."""
    from PIL import Image
    imgs = row.get("images", row.get("image"))
    if isinstance(imgs, list):
        if len(imgs) != 1:
            return None
        imgs = imgs[0]
    if isinstance(imgs, dict) and imgs.get("bytes"):
        try:
            return Image.open(io.BytesIO(imgs["bytes"]))
        except Exception:
            return None
    if hasattr(imgs, "convert"):
        return imgs
    return None


def collect_deepeyes(target_count, inspect=False):
    name = "deepeyes"
    cached = _load_fragment(name)
    if cached is not None:
        print(f"  [{name}] fragment exists ({len(cached)} samples). Skipping download.")
        return cached

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    shard_paths = [
        hf_hub_download(DEEPEYES_REPO, s, repo_type="dataset")
        for s in DEEPEYES_SHARDS
    ]

    # Pre-pick kept row indices per shard so the subsample stays uniform
    # across shards without materializing anything.
    pfs = [pq.ParquetFile(p) for p in shard_paths]
    totals = [pf.metadata.num_rows for pf in pfs]
    total = sum(totals)
    print(f"  [{name}] shard rows: {totals} (total {total})")
    if target_count and total > target_count:
        picked = set(random.Random(SEED + 1).sample(range(total), target_count))
    else:
        picked = None  # take everything

    save_dir = os.path.join(TRAIN_SPLIT, name)
    os.makedirs(save_dir, exist_ok=True)

    samples, shown = [], 0
    from collections import Counter
    src_counter = Counter()
    offset = 0
    for pf, shard in zip(pfs, DEEPEYES_SHARDS):
        pbar = tqdm(desc=f"    {name}:{shard[:24]}", total=pf.metadata.num_rows)
        row_idx = offset
        # Small batches keep peak RSS low despite embedded images.
        for batch in pf.iter_batches(batch_size=8):
            for row in batch.to_pylist():
                cur = row_idx
                row_idx += 1
                pbar.update(1)
                if picked is not None and cur not in picked:
                    continue
                qa = _extract_qa_verl(row)
                if qa is None:
                    continue
                img = _row_single_image(row)
                if img is None:
                    continue
                fname = f"{name}_{cur:06d}.jpg"
                if not _save_jpg(img, save_dir, fname):
                    continue
                question, answer = qa
                samples.append(_make_sample(name, cur, fname, question, answer))
                src_counter[row.get("data_source", "?")] += 1
                if inspect and shown < 5:
                    _show(name, question, answer)
                    shown += 1
        pbar.close()
        offset += pf.metadata.num_rows

    print(f"  [{name}] kept {len(samples)} rows; data_source dist: {src_counter.most_common()}")
    _save_fragment(name, samples)
    return samples


# ---------------------------------------------------------------------------
#  VisualProbe  (train.json + raw jpgs on the hub)
# ---------------------------------------------------------------------------

def collect_visualprobe(target_count, inspect=False):
    name = "visualprobe"
    cached = _load_fragment(name)
    if cached is not None:
        print(f"  [{name}] fragment exists ({len(cached)} samples). Skipping download.")
        return cached

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    from tqdm import tqdm
    import shutil

    meta_path = hf_hub_download(VISUALPROBE_REPO, "train.json", repo_type="dataset")
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  [{name}] train.json: {len(meta)} rows")

    if target_count and len(meta) > target_count:
        meta = random.Random(SEED + 2).sample(meta, target_count)

    save_dir = os.path.join(TRAIN_SPLIT, name)
    os.makedirs(save_dir, exist_ok=True)

    # Fetch each image with a plain hf_hub_download call (no tree listing).
    # snapshot_download's recursive "list repo tree" API returns pagination
    # links that point at the real huggingface.co even when HF_ENDPOINT is a
    # mirror — hf-mirror.com doesn't fully proxy that endpoint — so every
    # retry there hits the blocked domain no matter how HF_ENDPOINT is set.
    # hf_hub_download builds file URLs directly from HF_ENDPOINT + filename,
    # sidestepping tree listing entirely.
    def _fetch(rel):
        for attempt in range(3):
            try:
                return hf_hub_download(VISUALPROBE_REPO, rel, repo_type="dataset")
            except HfHubHTTPError as e:
                print(f"    ! fetch {rel} attempt {attempt + 1} failed: {e}")
        return None

    samples, shown = [], 0
    for item in tqdm(meta, desc=f"    {name}"):
        imgs = item.get("images") or []
        if len(imgs) != 1:
            continue
        question = _clean_question(item.get("problem"))
        answer = _valid_short_answer(item.get("solution"))
        if not question or not answer:
            continue
        # e.g. "VisualProbe_train/data/visual_probe_train_7.jpg" -> "data/..."
        rel = imgs[0].split("/", 1)[1] if "/" in imgs[0] else imgs[0]
        fname = os.path.basename(rel)
        dst = os.path.join(save_dir, fname)
        if not os.path.exists(dst):
            src = _fetch(rel)
            if src is None:
                continue
            shutil.copy2(src, dst)
        sample_id = item.get("doc_id", os.path.splitext(fname)[0])
        samples.append(_make_sample(name, sample_id, fname, question, answer))
        if inspect and shown < 5:
            _show(name, question, answer)
            shown += 1

    print(f"  [{name}] kept {len(samples)} rows")
    _save_fragment(name, samples)
    return samples


# ---------------------------------------------------------------------------
#  Merge (phase 2)
# ---------------------------------------------------------------------------

def prefix_stats(samples):
    from collections import Counter
    c = Counter()
    for s in samples:
        img = s.get("image")
        c[img.split("/")[0] if img else "<text-only>"] += 1
    return c


def merge(args):
    print("\n== merge phase ==")
    with open(BASE_JSON) as f:
        merged = json.load(f)
    print(f"  base ivcap: {len(merged)}")

    take = [p for p in HD251K_TAKE_PREFIXES if not (p == "ai2d" and args.no_ai2d)]
    with open(HD251K_JSON) as f:
        hd = json.load(f)
    ocr = [s for s in hd if s.get("image", "").split("/")[0] in take]
    print(f"  from hd251k ({'+'.join(take)}): {len(ocr)}")
    del hd
    merged.extend(ocr)

    for name in ["stvqa", "deepeyes", "visualprobe"]:
        frag = _load_fragment(name)
        if frag is None:
            print(f"  [WARN] fragment missing: {name} (run without --skip_downloads, "
                  f"or copy extra_0812/{name}.json over). Continuing without it.")
            continue
        merged.extend(frag)
        print(f"  fragment {name}: {len(frag)}")

    # id de-dup (prefixes shouldn't collide, but be safe). The ivcap base has
    # NO "id" field at all (verified: 0 occurrences across 369419 entries) —
    # treat missing-id entries as always-unique rather than crashing/
    # colliding on a shared sentinel key.
    seen, unique = set(), []
    for idx, s in enumerate(merged):
        key = s.get("id", f"__noid_{idx}__")
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    if len(unique) < len(merged):
        print(f"  de-dup: {len(merged)} -> {len(unique)}")
    merged = unique

    # Verify images exist on disk — but ONLY for the prefixes this script
    # adds. Base ivcap entries are trusted as-is: on the training cluster the
    # OSS train_split does not contain e.g. the sa1b images (they live behind
    # a local symlink farm), and dropping those 50k entries here would be a
    # silent disaster.
    check_prefixes = set(HD251K_TAKE_PREFIXES) | {"stvqa", "deepeyes", "visualprobe"}
    missing_by_prefix = {}
    kept = []
    for s in merged:
        img = s.get("image", "")
        prefix = img.split("/")[0] if img else ""
        if prefix in check_prefixes and not os.path.exists(os.path.join(TRAIN_SPLIT, img)):
            missing_by_prefix[prefix] = missing_by_prefix.get(prefix, 0) + 1
            continue
        kept.append(s)
    if missing_by_prefix:
        print(f"  [WARN] dropped samples with missing images: {missing_by_prefix} "
              f"(image dirs not in $SFT_DIR/train_split yet?)")
    merged = kept

    random.Random(SEED).shuffle(merged)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(merged, f, ensure_ascii=False)

    print(f"\n  TOTAL {len(merged)} -> {OUTPUT_JSON}")
    for k, v in prefix_stats(merged).most_common():
        print(f"    {k:14s} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_downloads", action="store_true",
                    help="merge only, use existing extra_0812/ fragments")
    ap.add_argument("--download_only", action="store_true",
                    help="downloads only, skip the merge phase")
    ap.add_argument("--only", type=str, default="stvqa,deepeyes,visualprobe",
                    help="comma-separated subset of downloads to run")
    ap.add_argument("--inspect", action="store_true",
                    help="print a few converted samples per new dataset")
    ap.add_argument("--no_ai2d", action="store_true",
                    help="exclude ai2d (it is the lmms-lab TEST split)")
    ap.add_argument("--stvqa_count", type=int, default=26000)
    ap.add_argument("--deepeyes_count", type=int, default=25000)
    ap.add_argument("--visualprobe_count", type=int, default=None,
                    help="default: take all (~5.7k)")
    args = ap.parse_args()

    if not args.skip_downloads:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
        print(f"== download phase ({', '.join(sorted(only))}) ==")
        for label, fn in [
            ("stvqa", lambda: collect_stvqa(args.stvqa_count, args.inspect)),
            ("deepeyes", lambda: collect_deepeyes(args.deepeyes_count, args.inspect)),
            ("visualprobe", lambda: collect_visualprobe(args.visualprobe_count, args.inspect)),
        ]:
            if label not in only:
                continue
            try:
                fn()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  [{label}] ERROR: {type(e).__name__}: {e}. Skipping this dataset.")
                import traceback
                traceback.print_exc()

    if not args.download_only:
        merge(args)


if __name__ == "__main__":
    main()
