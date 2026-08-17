#!/usr/bin/env python3
"""Build the 0817 "general + visual search" SFT mix (v2 of the 0812 mix).

Goal: KEEP needle-in-haystack scores (vstar/HRBench — guarded by raising the
sa1b/visualprobe anchor share to ~16.7%) while ATTACKING the general-task gap
vs base Qwen2.5-VL (MMBench/SEED/MME/SQA) with high-quality, high-res data.

Changes vs llava_hr_ocr_vs_0812.json (see chat 0817 for the full rationale):
  needle    sa1b 50k -> 75k (x1.5 dup), visualprobe 5.7k -> 11.5k (x2 dup),
            deepeyes 24.3k kept
  removed   synthdog 40k (synthetic, low quality), hd251k chartqa/textvqa/
            ai2d 53k (proven zero OCR gain in 0812), ocr_vqa 80k -> 30k,
            stvqa 26k -> 5k
  added     densefusion 100k  (DenseFusion-4V-100K: GPT-4V hyper-detailed
                               captions on high-res LAION images)
            allava      50k   (ALLaVA-Instruct-LAION-4V: GPT-4V instruct QA)
            aokvqa      ~17k  (A-OKVQA train, MC + letter answers)
            scienceqa   ~6.3k (ScienceQA train, image split, MC letters)

Run on the training cluster (OSS mounted):

  export SFT_DIR=/data/oss_bucket_0/wangziyi/models_data
  export HF_HOME=<big local disk>/hf     # zips are large (~130GB total)
  python construct_sft_0817.py --download_only          # stage 1 (overnight)
  python construct_sft_0817.py --skip_downloads         # stage 2 (merge)
  python construct_sft_0817.py --only allava --inspect  # debug one dataset

Outputs $SFT_DIR/llava_hr_gen_vs_0817.json + images under
$SFT_DIR/train_split/{densefusion,allava,aokvqa,scienceqa}/.
Reuses 0812 fragments (extra_0812/{stvqa,deepeyes,visualprobe}.json) as-is.
"""

import argparse
import functools
import io
import json
import os
import random
import zipfile

print = functools.partial(print, flush=True)  # tee-safe: no silent buffering

SFT_DIR = os.environ.get("SFT_DIR", "/root/autodl-tmp/models_data/sft_data")
TRAIN_SPLIT = os.path.join(SFT_DIR, "train_split")
EXTRA_DIR = os.path.join(SFT_DIR, "extra_0817")
EXTRA_0812_DIR = os.path.join(SFT_DIR, "extra_0812")

BASE_JSON = os.path.join(SFT_DIR, "llava_hr_essential_sa1b_ivcap.json")
OUTPUT_JSON = os.path.join(SFT_DIR, "llava_hr_gen_vs_0817.json")

SEED = 42

# -- mix knobs ---------------------------------------------------------------
SA1B_UPSAMPLE = 1.5          # 50k -> 75k
VISUALPROBE_UPSAMPLE = 2     # 5.7k -> 11.5k
OCRVQA_KEEP = 30_000         # from 80k
STVQA_KEEP = 5_000           # from 26k fragment
DROP_PREFIXES = {"synthdog"}

DENSEFUSION_COUNT = 100_000
ALLAVA_COUNT = 50_000
# aokvqa / scienceqa: take the full train (image) splits, no cap needed.

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

DENSEFUSION_REPO = "BAAI/DenseFusion-1M"
ALLAVA_REPO = "FreedomIntelligence/ALLaVA-4V"
AOKVQA_REPO = "HuggingFaceM4/A-OKVQA"
SCIENCEQA_REPO = "derek-thomas/ScienceQA"

MC_SUFFIX = "Answer with the option's letter from the given choices directly."
CAPTION_PROMPTS = [
    "Describe this image in detail.",
    "Provide a thorough description of the image.",
    "What is happening in this image? Describe it comprehensively.",
    "Give a detailed account of everything visible in this image.",
    "Write a comprehensive and detailed caption for this image.",
]


# ---------------------------------------------------------------------------
#  Shared helpers (same conventions as construct_sft_0812.py)
# ---------------------------------------------------------------------------

def _fragment_path(name):
    return os.path.join(EXTRA_DIR, f"{name}.json")


def _load_fragment(name, base_dir=None):
    p = os.path.join(base_dir, f"{name}.json") if base_dir else _fragment_path(name)
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


def _mc_block(question, choices, hint=None):
    letters = "ABCDEFGH"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    parts = []
    if hint:
        parts.append(hint.strip())
    parts.append(question.strip())
    parts.append(opts)
    parts.append(MC_SUFFIX)
    return "\n".join(parts)


def _show(name, question, answer):
    print(f"\n  [{name} sample] Q: {question[:300]}\n{' ' * (len(name) + 12)}A: {str(answer)[:200]}")


def _hf_download(repo, filename, repo_type="dataset", retries=3):
    from huggingface_hub import hf_hub_download
    for attempt in range(retries):
        try:
            return hf_hub_download(repo, filename, repo_type=repo_type)
        except Exception as e:
            print(f"    ! download {filename} attempt {attempt + 1} failed: {type(e).__name__}: {e}")
    raise RuntimeError(f"failed to download {repo}/{filename} after {retries} tries")


def _list_repo_files(repo):
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(repo, repo_type="dataset")


def _extract_zip(zip_path, dest_dir, want_exts=(".jpg", ".jpeg", ".png", ".webp")):
    """Flatten-extract image members; returns list of extracted basenames."""
    os.makedirs(dest_dir, exist_ok=True)
    names = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if not m.endswith("/") and os.path.splitext(m)[1].lower() in want_exts]
        from tqdm import tqdm
        for m in tqdm(members, desc=f"    unzip {os.path.basename(zip_path)}"):
            base = os.path.basename(m)
            out = os.path.join(dest_dir, base)
            names.append(base)
            if os.path.exists(out):
                continue
            with zf.open(m) as src, open(out, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
    return names


# ---------------------------------------------------------------------------
#  A-OKVQA (train, MC with rationales; images embedded in parquet)
# ---------------------------------------------------------------------------

def collect_aokvqa(inspect=False):
    name = "aokvqa"
    cached = _load_fragment(name)
    if cached is not None:
        print(f"  [{name}] fragment exists ({len(cached)} samples). Skipping download.")
        return cached

    from datasets import load_dataset
    from tqdm import tqdm

    ds = load_dataset(AOKVQA_REPO, split="train")
    print(f"  [{name}] loaded {AOKVQA_REPO} train ({len(ds)} rows)")
    save_dir = os.path.join(TRAIN_SPLIT, name)
    os.makedirs(save_dir, exist_ok=True)

    letters = "ABCDEFGH"
    samples, shown = [], 0
    for idx, row in enumerate(tqdm(ds, desc=f"    {name}")):
        img = row.get("image")
        choices = row.get("choices") or []
        ci = row.get("correct_choice_idx")
        q = row.get("question")
        if img is None or not q or not choices or ci is None or ci >= len(choices):
            continue
        fname = f"{name}_{idx:06d}.jpg"
        if not _save_jpg(img, save_dir, fname):
            continue
        question = _mc_block(q, choices)
        answer = letters[ci]
        samples.append(_make_sample(name, idx, fname, question, answer))
        if inspect and shown < 3:
            _show(name, question, answer)
            shown += 1

    print(f"  [{name}] kept {len(samples)} samples")
    _save_fragment(name, samples)
    return samples


# ---------------------------------------------------------------------------
#  ScienceQA (train split, image-bearing rows only; MC letters)
# ---------------------------------------------------------------------------

def collect_scienceqa(inspect=False):
    name = "scienceqa"
    cached = _load_fragment(name)
    if cached is not None:
        print(f"  [{name}] fragment exists ({len(cached)} samples). Skipping download.")
        return cached

    from datasets import load_dataset
    from tqdm import tqdm

    ds = load_dataset(SCIENCEQA_REPO, split="train")
    print(f"  [{name}] loaded {SCIENCEQA_REPO} train ({len(ds)} rows)")
    save_dir = os.path.join(TRAIN_SPLIT, name)
    os.makedirs(save_dir, exist_ok=True)

    letters = "ABCDEFGH"
    samples, shown = [], 0
    for idx, row in enumerate(tqdm(ds, desc=f"    {name}")):
        img = row.get("image")
        choices = row.get("choices") or []
        ans = row.get("answer")
        q = row.get("question")
        if img is None or not q or not choices or ans is None or ans >= len(choices):
            continue
        fname = f"{name}_{idx:06d}.jpg"
        if not _save_jpg(img, save_dir, fname):
            continue
        question = _mc_block(q, choices, hint=row.get("hint") or None)
        answer = letters[ans]
        samples.append(_make_sample(name, idx, fname, question, answer))
        if inspect and shown < 3:
            _show(name, question, answer)
            shown += 1

    print(f"  [{name}] kept {len(samples)} samples (image-bearing train rows)")
    _save_fragment(name, samples)
    return samples


# ---------------------------------------------------------------------------
#  ALLaVA-Instruct-LAION-4V (GPT-4V instruct QA; images in repo zips)
# ---------------------------------------------------------------------------

def collect_allava(target_count, inspect=False):
    name = "allava"
    cached = _load_fragment(name)
    if cached is not None:
        print(f"  [{name}] fragment exists ({len(cached)} samples). Skipping download.")
        return cached

    meta_path = _hf_download(ALLAVA_REPO, "allava_laion/ALLaVA-Instruct-LAION-4V.json")
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  [{name}] instruct json: {len(meta)} rows")

    zips = sorted(f for f in _list_repo_files(ALLAVA_REPO)
                  if f.startswith("allava_laion/") and f.endswith(".zip"))
    print(f"  [{name}] repo image zips: {zips}")

    save_dir = os.path.join(TRAIN_SPLIT, name)
    os.makedirs(save_dir, exist_ok=True)

    # Download + extract zips until we have ~15% more images than needed
    # (some meta rows will not match). Zips are huge — one at a time.
    available = set(os.listdir(save_dir)) if os.path.isdir(save_dir) else set()
    for z in zips:
        if len(available) >= target_count * 1.15:
            break
        print(f"  [{name}] fetching {z} (have {len(available)} images)")
        zp = _hf_download(ALLAVA_REPO, z)
        available.update(_extract_zip(zp, save_dir))
    print(f"  [{name}] images available: {len(available)}")

    rng = random.Random(SEED + 3)
    rng.shuffle(meta)
    samples, shown = [], 0
    for row in meta:
        if len(samples) >= target_count:
            break
        img = os.path.basename(row.get("image", ""))
        conv = row.get("conversations") or []
        if not img or img not in available or len(conv) < 2:
            continue
        q = conv[0].get("value", "").replace("<image>", "").strip()
        a = conv[1].get("value", "").strip()
        if not q or not a:
            continue
        sid = row.get("id", img.rsplit(".", 1)[0])
        samples.append(_make_sample(name, sid, img, q, a))
        if inspect and shown < 3:
            _show(name, q, a)
            shown += 1

    print(f"  [{name}] kept {len(samples)} samples")
    _save_fragment(name, samples)
    return samples


# ---------------------------------------------------------------------------
#  DenseFusion-4V-100K (GPT-4V dense captions on high-res LAION; repo zips)
# ---------------------------------------------------------------------------

def collect_densefusion(target_count, inspect=False):
    name = "densefusion"
    cached = _load_fragment(name)
    if cached is not None:
        print(f"  [{name}] fragment exists ({len(cached)} samples). Skipping download.")
        return cached

    files = _list_repo_files(DENSEFUSION_REPO)
    jsonl_cands = [f for f in files if f.lower().endswith(".jsonl") and "4v-100k" in f.lower()]
    if not jsonl_cands:
        raise RuntimeError(f"no 4V-100k jsonl found in {DENSEFUSION_REPO}; files: {files[:20]}")
    meta_path = _hf_download(DENSEFUSION_REPO, jsonl_cands[0])

    rows = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"  [{name}] {jsonl_cands[0]}: {len(rows)} rows; keys: {sorted(rows[0].keys())}")

    def _get(row, *keys):
        for k in keys:
            if row.get(k):
                return row[k]
        return None

    zips = sorted(f for f in files
                  if f.endswith(".zip") and "4v-100k" in f.lower() and f.startswith("images/"))
    if not zips:  # fall back to the 1M zips if the 100K set has none
        zips = sorted(f for f in files if f.endswith(".zip") and f.startswith("images/"))
    print(f"  [{name}] image zips to fetch: {len(zips)} ({zips[:3]}...)")

    save_dir = os.path.join(TRAIN_SPLIT, name)
    os.makedirs(save_dir, exist_ok=True)
    available = set(os.listdir(save_dir)) if os.path.isdir(save_dir) else set()
    for z in zips:
        if len(available) >= target_count * 1.1:
            break
        print(f"  [{name}] fetching {z} (have {len(available)} images)")
        zp = _hf_download(DENSEFUSION_REPO, z)
        available.update(_extract_zip(zp, save_dir))
    print(f"  [{name}] images available: {len(available)}")

    # Map basename (with and without extension) -> actual filename on disk.
    by_stem = {}
    for fname in available:
        by_stem[fname] = fname
        by_stem[os.path.splitext(fname)[0]] = fname

    rng = random.Random(SEED + 4)
    rng.shuffle(rows)
    samples, shown = [], 0
    for row in rows:
        if len(samples) >= target_count:
            break
        img_key = _get(row, "image_id", "image", "id", "img_id")
        cap = _get(row, "caption", "description", "text", "conversations")
        if isinstance(cap, list):  # llava-style conversations fallback
            cap = next((c.get("value") for c in cap if c.get("from") == "gpt"), None)
        if not img_key or not cap or not isinstance(cap, str):
            continue
        stem = os.path.basename(str(img_key))
        fname = by_stem.get(stem) or by_stem.get(os.path.splitext(stem)[0])
        if not fname:
            continue
        q = CAPTION_PROMPTS[len(samples) % len(CAPTION_PROMPTS)]
        sid = os.path.splitext(fname)[0]
        samples.append(_make_sample(name, sid, fname, q, cap.strip()))
        if inspect and shown < 3:
            _show(name, q, cap)
            shown += 1

    print(f"  [{name}] kept {len(samples)} samples")
    _save_fragment(name, samples)
    return samples


# ---------------------------------------------------------------------------
#  Merge
# ---------------------------------------------------------------------------

def prefix_stats(samples):
    from collections import Counter
    c = Counter()
    for s in samples:
        img = s.get("image")
        c[img.split("/")[0] if img else "<text-only>"] += 1
    return c


def merge(args):
    rng = random.Random(SEED)
    print("\n== merge phase ==")
    with open(BASE_JSON) as f:
        base = json.load(f)
    print(f"  base ivcap: {len(base)}")

    merged, ocrvqa_pool, sa1b_pool = [], [], []
    for s in base:
        prefix = s.get("image", "").split("/")[0]
        if prefix in DROP_PREFIXES:
            continue
        if prefix == "ocr_vqa":
            ocrvqa_pool.append(s)
        elif prefix == "sa1b":
            sa1b_pool.append(s)
            merged.append(s)
        else:
            merged.append(s)

    keep_ocr = rng.sample(ocrvqa_pool, min(OCRVQA_KEEP, len(ocrvqa_pool)))
    merged.extend(keep_ocr)
    print(f"  ocr_vqa: {len(ocrvqa_pool)} -> {len(keep_ocr)};  synthdog dropped")

    n_dup = int(len(sa1b_pool) * (SA1B_UPSAMPLE - 1.0))
    sa1b_dup = [dict(s) for s in rng.sample(sa1b_pool, min(n_dup, len(sa1b_pool)))]
    merged.extend(sa1b_dup)
    print(f"  sa1b: {len(sa1b_pool)} + {len(sa1b_dup)} dup = {len(sa1b_pool) + len(sa1b_dup)}")

    # -- 0812 fragments (reused) --
    stvqa = _load_fragment("stvqa", EXTRA_0812_DIR)
    if stvqa:
        keep = rng.sample(stvqa, min(STVQA_KEEP, len(stvqa)))
        merged.extend(keep)
        print(f"  stvqa: {len(stvqa)} -> {len(keep)}")
    else:
        print("  [WARN] extra_0812/stvqa.json missing — continuing without stvqa")

    deepeyes = _load_fragment("deepeyes", EXTRA_0812_DIR)
    if deepeyes:
        merged.extend(deepeyes)
        print(f"  deepeyes: {len(deepeyes)} (kept in full)")
    else:
        print("  [WARN] extra_0812/deepeyes.json missing — continuing without deepeyes")

    vp = _load_fragment("visualprobe", EXTRA_0812_DIR)
    if vp:
        merged.extend(vp)
        for k in range(1, VISUALPROBE_UPSAMPLE):
            for s in vp:
                d = dict(s)
                d["id"] = f"{s.get('id', 'visualprobe')}_dup{k}"
                merged.append(d)
        print(f"  visualprobe: {len(vp)} x{VISUALPROBE_UPSAMPLE} = {len(vp) * VISUALPROBE_UPSAMPLE}")
    else:
        print("  [WARN] extra_0812/visualprobe.json missing — continuing without visualprobe")

    # -- 0817 fragments (new) --
    for nm in ["densefusion", "allava", "aokvqa", "scienceqa"]:
        frag = _load_fragment(nm)
        if frag is None:
            print(f"  [WARN] fragment missing: {nm} (run download phase first). Continuing without it.")
            continue
        merged.extend(frag)
        print(f"  fragment {nm}: {len(frag)}")

    # id de-dup. Missing-id entries (the whole ivcap base) and intentional
    # duplicates (sa1b upsample copies share no id; visualprobe dups carry a
    # _dupN suffix) are all treated as unique via the per-index fallback key.
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

    # Verify images exist on disk — only for prefixes this script touches.
    # (tqdm here: on the OSS FUSE mount each os.path.exists is a network
    # round-trip; last time this loop looked hung for 25 minutes.)
    from tqdm import tqdm
    check_prefixes = {"stvqa", "deepeyes", "visualprobe",
                      "densefusion", "allava", "aokvqa", "scienceqa"}
    missing_by_prefix = {}
    kept = []
    for s in tqdm(merged, desc="  verify images"):
        img = s.get("image", "")
        prefix = img.split("/")[0] if img else ""
        if prefix in check_prefixes and not os.path.exists(os.path.join(TRAIN_SPLIT, img)):
            missing_by_prefix[prefix] = missing_by_prefix.get(prefix, 0) + 1
            continue
        kept.append(s)
    if missing_by_prefix:
        print(f"  [WARN] dropped samples with missing images: {missing_by_prefix}")
    merged = kept

    random.Random(SEED).shuffle(merged)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(merged, f, ensure_ascii=False)

    print(f"\n  TOTAL {len(merged)} -> {OUTPUT_JSON}")
    for k, v in prefix_stats(merged).most_common():
        print(f"    {k:14s} {v}")
    needle = sum(v for k, v in prefix_stats(merged).items()
                 if k in ("sa1b", "visualprobe"))
    print(f"  needle anchor share (sa1b+visualprobe): {needle}/{len(merged)}"
          f" = {100 * needle / len(merged):.1f}%")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_downloads", action="store_true",
                    help="merge only (fragments must already exist)")
    ap.add_argument("--download_only", action="store_true",
                    help="download/convert fragments, skip the merge")
    ap.add_argument("--only", action="append", default=None,
                    choices=["densefusion", "allava", "aokvqa", "scienceqa"],
                    help="restrict download phase to these datasets")
    ap.add_argument("--inspect", action="store_true",
                    help="print a few converted samples per dataset")
    args = ap.parse_args()

    if not args.skip_downloads:
        only = set(args.only) if args.only else {"densefusion", "allava", "aokvqa", "scienceqa"}
        print(f"== download phase ({', '.join(sorted(only))}) ==")
        jobs = [
            ("aokvqa", lambda: collect_aokvqa(inspect=args.inspect)),
            ("scienceqa", lambda: collect_scienceqa(inspect=args.inspect)),
            ("allava", lambda: collect_allava(ALLAVA_COUNT, inspect=args.inspect)),
            ("densefusion", lambda: collect_densefusion(DENSEFUSION_COUNT, inspect=args.inspect)),
        ]
        for label, fn in jobs:
            if label not in only:
                continue
            try:
                fn()
            except Exception as e:
                print(f"  [{label}] ERROR: {type(e).__name__}: {e}. Skipping this dataset.")

    if not args.download_only:
        merge(args)


if __name__ == "__main__":
    main()
