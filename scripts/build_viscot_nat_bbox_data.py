#!/usr/bin/env python3
"""Visual-CoT answer-region bboxes on NATURAL images (the doc subsets are
handled by scripts/build_viscot_bbox_data.py, which joins on QA text against
images we already have).

Why: the 0919/0920 mini readout SFTs were trained on Visual-CoT DOCUMENT boxes
only; the readout learned to read text from HD (held-out oracle +2.5 / real
+2.0) but nothing transferred to V* (natural images, small-object attributes).
The full 0920 run needs natural-image samples whose answer lives in a box.

Visual-CoT natural-image metadata (deepcs233/Visual-CoT, metadata/*.jsonl; the
`image` field is the ORIGINAL dataset's file name, so images come from the
original sources -- the 139 GB cot_images tar on HF is not needed):

  subset      rows    imgs   native size   image source                 how we get the image
  openimages  43k     29k    1024x768      Open Images train            S3 per-id download (no auth)
  textcap     32k     16k    1024x768      Open Images train (TextCaps) S3 per-id download, same id space
  gqa         98k     54k    500x375       Visual Genome                local: train_split/gqa or vg
  v7w         30k     13k    500x375       Visual Genome (v7w_<id>.jpg) local: train_split/gqa or vg
  vsr         3.4k    1.8k   640x480       COCO 2017 (12-digit names)   local: train_split/coco/train2017
  flickr30k   136k    28k    500x375       Flickr30k                    local dir you provide (--local_dir)
  cub         10k     5k     500x375       CUB-200-2011                 local dir you provide (--local_dir)

RESOLUTION CAVEAT (read before mixing these in). The trainer only builds a
window for a bbox sample when HD carries dat_tf_min_hd_ratio (default 2.0) x
the LR pixels. SFT LR is clamped to [200704, 501760] px, HD = min(9 x LR,
native, 5017600). So:
  500x375  (gqa/v7w/flickr/cub) -> LR is UPscaled to 200k, HD = native 187k:
           ratio < 1, NO window ever. HD is pure redundancy on these images.
  640x480  (vsr)                -> ratio 0.6, same.
  1024x768 (openimages/textcap) -> LR 501760, HD 786k: ratio 1.57. Passes only
           with TF_MIN_HD_RATIO <= 1.5. HD has 1.25x the linear resolution of
           LR -- marginal for the readout, fine for offset supervision (the
           pull only needs a question -> location target).
The script prints this ratio per subset (--report) and can filter on it
(--min_hd_ratio). Default subsets are openimages,textcap for that reason.
For truly HD-necessary natural-image boxes use synth_hd_text (SA-1B 1500x2250
backgrounds, scripts/build_synth_hd_text_data.py) alongside these.

Output: LLaVA-format single-turn samples with a top-level ``bbox`` field
([x0, y0, x1, y1] as image fractions; several boxes -> union), the same
convention as build_viscot_bbox_data.py / build_tf_bbox_data.py; consumed by
Qwen2VLCoupledDATDataset. Image paths are relative to --image_root.

Usage (OSS cluster; images land physically on OSS, train_split is the symlink
farm on local disk):
  OSS=/data/oss_bucket_0/wangziyi/models_data
  python scripts/build_viscot_nat_bbox_data.py \
      --meta_dir ~/cache/viscot --subsets openimages,textcap \
      --download_dir $OSS/train_split/viscot_nat/openimages \
      --image_root ~/sft_data/train_split --workers 32 \
      --out $OSS/extra_0920/viscot_nat_bbox.json --heldout 300
  ln -sfn $OSS/train_split/viscot_nat ~/sft_data/train_split/viscot_nat
Local-join subsets (no download):
  python scripts/build_viscot_nat_bbox_data.py --subsets gqa,v7w,vsr \
      --image_root ~/sft_data/train_split --out .../viscot_nat_lowres_bbox.json
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_viscot_bbox_data import fetch_meta, union_box  # noqa: E402

# subset -> (metadata name, image source, default local search dirs under image_root)
SUBSETS = {
    "openimages": ("openimages", "s3", []),
    "textcap":    ("textcap", "s3", []),
    "gqa":        ("gqa", "local", ["gqa/images", "gqa", "vg/VG_100K", "vg/VG_100K_2"]),
    "v7w":        ("visual7w", "local", ["gqa/images", "gqa", "vg/VG_100K", "vg/VG_100K_2"]),
    "vsr":        ("vsr", "local", ["coco/train2017", "coco/val2017", "coco"]),
    "flickr30k":  ("flickr30k", "local", ["flickr30k/images", "flickr30k"]),
    "cub":        ("cub", "local", ["cub/images", "CUB_200_2011/images", "cub"]),
}
S3_URL = "https://open-images-dataset.s3.amazonaws.com/train/{name}"

# trainer geometry (legacy LR-first path of the 0915 SFT script)
LR_MIN, LR_MAX, HD_MAX, HR_SCALE = 200704, 501760, 5017600, 3


def hd_lr_ratio(w, h):
    native = float(w * h)
    lr = min(max(native, LR_MIN), LR_MAX)
    hd = min(lr * HR_SCALE ** 2, native, HD_MAX)
    return hd / lr


def local_name(subset, image):
    """metadata image name -> candidate file names in the local dirs."""
    if subset == "v7w" and image.startswith("v7w_"):
        return [image[len("v7w_"):], image]
    return [image]


def index_dir(root):
    """file name -> relative path, one scandir per directory (no recursion:
    the candidate dirs are flat)."""
    idx = {}
    if not os.path.isdir(root):
        return idx
    with os.scandir(root) as it:
        for e in it:
            if e.is_file() or e.is_symlink():
                idx[e.name] = e.name
    return idx


def download_one(name, dst_dir, retries=3, timeout=60):
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return name, "cached"
    url = S3_URL.format(name=name)
    tmp = dst + ".part"
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(tmp) == 0:
                raise IOError("empty body")
            os.replace(tmp, dst)
            return name, "ok"
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            if "404" in err:
                break
            time.sleep(1.5 * (k + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    return name, f"fail:{err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", default="/tmp/viscot")
    ap.add_argument("--subsets", default="openimages,textcap")
    ap.add_argument("--image_root", required=True,
                    help="training image root (train_split); output paths are relative to it")
    ap.add_argument("--download_dir", default=None,
                    help="where S3 images are written (default <image_root>/viscot_nat/openimages). "
                         "On the OSS cluster point this at the OSS train_split and symlink it into "
                         "the local farm; must resolve to <image_root>/<rel> for the output paths.")
    ap.add_argument("--download_rel", default="viscot_nat/openimages",
                    help="path of --download_dir relative to --image_root (used in the output json)")
    ap.add_argument("--no_download", action="store_true",
                    help="only use S3 images already present in --download_dir")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max_images", type=int, default=0,
                    help=">0: for S3 subsets, download at most this many (random) images; "
                         "rows on other images are dropped as missing_image")
    ap.add_argument("--local_dir", action="append", default=[],
                    help="override/add local search dirs: subset=rel/dir[,rel/dir2] (relative to image_root)")
    ap.add_argument("--max_box_frac", type=float, default=0.25,
                    help="drop samples whose union bbox covers more than this fraction of the image")
    ap.add_argument("--min_box_frac", type=float, default=0.0)
    ap.add_argument("--min_hd_ratio", type=float, default=0.0,
                    help="drop samples whose simulated HD/LR pixel ratio (trainer geometry) is below "
                         "this; the trainer's own guard is dat_tf_min_hd_ratio (default 2.0)")
    ap.add_argument("--max_per_image", type=int, default=0,
                    help=">0: keep at most this many QA pairs per image (gqa/flickr have 2-5 per image)")
    ap.add_argument("--max_samples", type=int, default=0, help=">0: cap per subset (after filters, random)")
    ap.add_argument("--heldout", type=int, default=0,
                    help=">0: split this many samples into <out>.heldout.json (for probe --dataset synth)")
    ap.add_argument("--verify", type=int, default=0,
                    help="1 = open every kept image with PIL and check its size against the metadata")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    extra_dirs = collections.defaultdict(list)
    for spec in args.local_dir:
        k, v = spec.split("=", 1)
        extra_dirs[k].extend(v.split(","))

    dl_dir = args.download_dir or os.path.join(args.image_root, args.download_rel)
    out, held, report = [], [], collections.OrderedDict()
    for sub in [s for s in args.subsets.split(",") if s]:
        if sub not in SUBSETS:
            sys.exit(f"unknown subset {sub}; choose from {sorted(SUBSETS)}")
        meta_name, source, dirs = SUBSETS[sub]
        dirs = extra_dirs.get(sub, []) + dirs
        rows = fetch_meta(args.meta_dir, meta_name)
        c = collections.Counter(kept=0)
        sizes = collections.Counter()
        ratios = []

        # ---- resolve images ----
        # name -> relative path under image_root
        resolved = {}
        if source == "s3":
            os.makedirs(dl_dir, exist_ok=True)
            names = sorted({r["image"] for r in rows})
            have = index_dir(dl_dir)
            if args.max_images and len(names) > args.max_images:
                # keep what is already on disk first, then fill up at random
                present = [n for n in names if n in have]
                rest = [n for n in names if n not in have]
                rng.shuffle(rest)
                names = present + rest[:max(0, args.max_images - len(present))]
            todo = [n for n in names if n not in have]
            print(f"  [{sub}] {len(names)} images, {len(names) - len(todo)} present in {dl_dir}, "
                  f"{len(todo)} to download" + (" (skipped: --no_download)" if args.no_download else ""))
            if todo and not args.no_download:
                t0, done = time.time(), collections.Counter()
                with cf.ThreadPoolExecutor(args.workers) as ex:
                    for i, (n, st) in enumerate(ex.map(lambda n: download_one(n, dl_dir), todo), 1):
                        done[st.split(":")[0]] += 1
                        if st.startswith("fail") and done["fail"] <= 5:
                            print(f"    {n}: {st}")
                        if i % 2000 == 0 or i == len(todo):
                            el = time.time() - t0
                            print(f"    {i}/{len(todo)} ({el:.0f}s, {i / max(el, 1e-6):.1f} img/s) {dict(done)}",
                                  flush=True)
                have = index_dir(dl_dir)
            for n in names:
                if n in have:
                    resolved[n] = os.path.join(args.download_rel, have[n])
        else:
            idx = {}
            for d in dirs:
                root = os.path.join(args.image_root, d)
                got = index_dir(root)
                if got:
                    print(f"  [{sub}] {len(got)} files in {root}")
                for k, v in got.items():
                    idx.setdefault(k, os.path.join(d, v))
            if not idx:
                print(f"  [{sub}] WARN: none of {dirs} exist under {args.image_root}; "
                      f"pass --local_dir {sub}=<rel/dir>")
            for r in rows:
                for cand in local_name(sub, r["image"]):
                    if cand in idx:
                        resolved[r["image"]] = idx[cand]
                        break

        # ---- build samples ----
        per_img = collections.Counter()
        rng.shuffle(rows)
        subset_out = []
        for r in rows:
            rel = resolved.get(r["image"])
            if rel is None:
                c["missing_image"] += 1
                continue
            W, H = float(r["width"]), float(r["height"])
            if W <= 0 or H <= 0 or not r.get("bboxs"):
                c["bad_meta"] += 1
                continue
            if args.verify:
                try:
                    from PIL import Image
                    with Image.open(os.path.join(args.image_root, rel)) as im:
                        w, h = im.size
                    if abs(w / h - W / H) > 0.02:
                        c["aspect_mismatch"] += 1
                        continue
                    W, H = float(w), float(h)
                except Exception:  # noqa: BLE001
                    c["unreadable_image"] += 1
                    continue
            b = union_box([list(map(float, bb)) for bb in r["bboxs"] if len(bb) == 4])
            fb = [min(max(v, 0.0), 1.0) for v in (b[0] / W, b[1] / H, b[2] / W, b[3] / H)]
            if fb[2] <= fb[0] or fb[3] <= fb[1]:
                c["degenerate_box"] += 1
                continue
            area = (fb[2] - fb[0]) * (fb[3] - fb[1])
            if area > args.max_box_frac:
                c["box_too_large"] += 1
                continue
            if area < args.min_box_frac:
                c["box_too_small"] += 1
                continue
            ratio = hd_lr_ratio(W, H)
            if ratio < args.min_hd_ratio:
                c["hd_ratio_low"] += 1
                continue
            if args.max_per_image and per_img[r["image"]] >= args.max_per_image:
                c["per_image_cap"] += 1
                continue
            per_img[r["image"]] += 1
            q = str(r["question"]).strip()
            a = str(r.get("answer", "")).strip()
            if not q or not a:
                c["empty_qa"] += 1
                continue
            sizes[(int(W), int(H))] += 1
            ratios.append(ratio)
            subset_out.append({
                "id": f"viscotnat_{sub}_{len(subset_out)}",
                "image": rel,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{q}"},
                    {"from": "gpt", "value": a},
                ],
                "bbox": [round(v, 4) for v in fb],
                "source": f"viscot_{sub}",
            })
            c["kept"] += 1
        if args.max_samples and len(subset_out) > args.max_samples:
            subset_out = subset_out[:args.max_samples]
            c["kept"] = len(subset_out)
        out.extend(subset_out)
        ratios.sort()
        rep = {
            "rows": len(rows), "kept": len(subset_out), "images": len(per_img),
            "top_sizes": sizes.most_common(3),
            "hd_lr_ratio_median": round(ratios[len(ratios) // 2], 2) if ratios else None,
            "frac_ratio_ge_2.0": round(sum(x >= 2.0 for x in ratios) / len(ratios), 3) if ratios else None,
            "frac_ratio_ge_1.5": round(sum(x >= 1.5 for x in ratios) / len(ratios), 3) if ratios else None,
            "drops": {k: v for k, v in c.items() if k != "kept"},
        }
        report[sub] = rep
        print(f"  [{sub}] {json.dumps(rep)}")

    if args.heldout and out:
        rng.shuffle(out)
        held, out = out[:args.heldout], out[args.heldout:]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"\nwrote {len(out)} bbox samples -> {args.out}")
    if held:
        hp = os.path.splitext(args.out)[0] + ".heldout.json"
        with open(hp, "w") as f:
            json.dump(held, f, ensure_ascii=False)
        print(f"wrote {len(held)} held-out -> {hp}")
    fr = sorted((s["bbox"][2] - s["bbox"][0]) * (s["bbox"][3] - s["bbox"][1]) for s in out)
    if fr:
        print(f"bbox area frac: median {fr[len(fr) // 2]:.4f}, p90 {fr[int(len(fr) * 0.9)]:.4f}, "
              f"<1%: {sum(v < 0.01 for v in fr) / len(fr):.0%}, <5%: {sum(v < 0.05 for v in fr) / len(fr):.0%}")
    low = [s for s, r in report.items() if r["frac_ratio_ge_2.0"] is not None and r["frac_ratio_ge_2.0"] < 0.5]
    if low:
        print(f"NOTE: {low} mostly fail the trainer's default dat_tf_min_hd_ratio=2.0 -> their windows are "
              f"dropped at train time unless TF_MIN_HD_RATIO is lowered (1.5 keeps 1024x768 images).")


if __name__ == "__main__":
    main()
