#!/usr/bin/env python3
"""Attach Visual-CoT answer-region bboxes to the docvqa/infovqa/textvqa samples
we already have on disk (llava_hd251k.json + train_split/{docvqa,infovqa,textvqa}).

Why: DeepEyes-47k / VisualProbe releases carry NO bbox (checked 0916: schema is
prompt / images / reward_model / extra_info only). Visual-CoT (deepcs233/Visual-CoT,
NeurIPS'24 D&B) annotates 438k QA pairs with the region needed to answer; its
docvqa / infographicsvqa / textvqa metadata reuse the original QA pairs, so we
can join on (question, answer) text against llava_hd251k.json and never download
an image. Bboxes are OCR regions of the answer -> tiny (99% < 5% of the image)
on native-resolution documents (DocVQA ~1.6k x 2.1k, InfoVQA up to 5k tall):
HD really carries detail LR lacks, which is what teacher forcing / offset
supervision need.

Output: LLaVA-format single-turn samples with a top-level ``bbox`` field
([x0, y0, x1, y1] as fractions of the image), same convention as
scripts/build_tf_bbox_data.py, consumed by Qwen2VLCoupledDATDataset.

Metadata files (small, ~23MB total; HF_ENDPOINT mirror works):
  https://huggingface.co/datasets/deepcs233/Visual-CoT/resolve/main/metadata/
      {docvqa,infographicsvqa,textvqa}_cot_train.jsonl

Usage:
  python scripts/build_viscot_bbox_data.py \
      --meta_dir /tmp/viscot \
      --hd251k $SFT_DIR/llava_hd251k.json \
      --image_root $SFT_DIR/train_split \
      --out $SFT_DIR/extra_0916/viscot_bbox.json
"""
import argparse
import collections
import json
import os
import re
import shutil
import urllib.request

from PIL import Image

SUBSETS = {  # local prefix -> Visual-CoT metadata name
    "docvqa": "docvqa",
    "infovqa": "infographicsvqa",
    "textvqa": "textvqa",
}
META_URL = "{endpoint}/datasets/deepcs233/Visual-CoT/resolve/main/metadata/{name}_cot_train.jsonl"


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower().rstrip("?").strip())


def fetch_meta(meta_dir, name):
    path = os.path.join(meta_dir, f"{name}_cot_train.jsonl")
    if not os.path.exists(path):
        os.makedirs(meta_dir, exist_ok=True)
        rel = f"metadata/{name}_cot_train.jsonl"
        # hf_hub_download follows the mirror's redirects the way the cluster
        # allows (all previous dataset pulls went through it); plain urllib got
        # a 403 after the 302 on the OSS cluster.
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            src = hf_hub_download("deepcs233/Visual-CoT", rel, repo_type="dataset")
            shutil.copyfile(src, path)
            print(f"  fetched {rel} via huggingface_hub")
        except Exception as e:
            print(f"  huggingface_hub failed ({e!r}); falling back to urllib")
            endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
            url = META_URL.format(endpoint=endpoint, name=name)
            print(f"  downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f)
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def union_box(bboxs):
    xs0, ys0, xs1, ys1 = zip(*bboxs)
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", default="/tmp/viscot")
    ap.add_argument("--hd251k", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--max_box_frac", type=float, default=0.25,
                    help="drop samples whose (union) bbox covers more than this "
                         "fraction of the image -- no point forcing those")
    ap.add_argument("--aspect_tol", type=float, default=0.02,
                    help="reject a join if local image aspect differs from the "
                         "metadata width/height by more than this")
    ap.add_argument("--check_dims", type=int, default=1,
                    help="open every matched image to verify aspect (slow-ish; "
                         "set 0 to trust the join)")
    ap.add_argument("--mix", default=None,
                    help="optional base training JSON (e.g. llava_hr_gen_vs_0817.json); "
                         "if given, --mix_out gets base + bbox samples, shuffled")
    ap.add_argument("--mix_out", default=None)
    ap.add_argument("--mix_repeat", type=int, default=1,
                    help="repeat the bbox samples this many times in the mix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"loading {args.hd251k}")
    hd = json.load(open(args.hd251k))
    out, stats = [], collections.OrderedDict()
    for pre in args.subsets.split(","):
        name = SUBSETS[pre]
        loc = collections.defaultdict(set)
        by_qa = collections.defaultdict(list)
        for x in hd:
            img = str(x.get("image", ""))
            if not img.startswith(pre + "/"):
                continue
            conv = x["conversations"]
            q = norm(conv[0]["value"].replace("<image>", ""))
            a = norm(conv[1]["value"])
            loc[(q, a)].add(img)
            loc[(q,)].add(img)
            by_qa[(q, a)].append(x)
        rows = fetch_meta(args.meta_dir, name)
        c = collections.Counter()
        for r in rows:
            q, a = norm(r["question"]), norm(r["answer"])
            cands = loc.get((q, a)) or loc.get((q,)) or set()
            if not cands:
                c["no_match"] += 1
                continue
            if len(cands) > 1:
                c["ambiguous"] += 1
                continue
            img = next(iter(cands))
            W, H = float(r["width"]), float(r["height"])
            if args.check_dims:
                try:
                    with Image.open(os.path.join(args.image_root, img)) as im:
                        w, h = im.size
                except Exception:
                    c["missing_image"] += 1
                    continue
                if abs(w / h - W / H) > args.aspect_tol:
                    c["aspect_mismatch"] += 1
                    continue
            b = union_box(r["bboxs"])
            fb = [b[0] / W, b[1] / H, b[2] / W, b[3] / H]
            fb = [min(max(v, 0.0), 1.0) for v in fb]
            if fb[2] <= fb[0] or fb[3] <= fb[1]:
                c["degenerate_box"] += 1
                continue
            if (fb[2] - fb[0]) * (fb[3] - fb[1]) > args.max_box_frac:
                c["box_too_large"] += 1
                continue
            # keep the original QA turn (it already matches the metadata answer)
            src = by_qa.get((q, a))
            question = r["question"].strip()
            answer = (src[0]["conversations"][1]["value"] if src else r["answer"]).strip()
            out.append({
                "id": f"viscot_{pre}_{len(out)}",
                "image": img,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
                "bbox": [round(v, 4) for v in fb],
                "source": f"viscot_{name}",
            })
            c["kept"] += 1
        stats[pre] = dict(c)
        print(f"  [{pre} <- {name}] meta rows {len(rows)}: {dict(c)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    fr = [(s["bbox"][2] - s["bbox"][0]) * (s["bbox"][3] - s["bbox"][1]) for s in out]
    fr.sort()
    print(f"\nwrote {len(out)} bbox samples -> {args.out}")
    if fr:
        print(f"bbox area frac: median {fr[len(fr)//2]:.4f}, p90 {fr[int(len(fr)*0.9)]:.4f}, "
              f"<1%: {sum(v < 0.01 for v in fr)/len(fr):.0%}, <5%: {sum(v < 0.05 for v in fr)/len(fr):.0%}")

    if args.mix:
        import random
        base = json.load(open(args.mix))
        mixed = list(base)
        for k in range(args.mix_repeat):
            for s in out:
                y = dict(s)
                if k:
                    y["id"] = f"{s['id']}_r{k}"
                mixed.append(y)
        random.Random(args.seed).shuffle(mixed)
        mix_out = args.mix_out or os.path.splitext(args.mix)[0] + "_viscot.json"
        with open(mix_out, "w") as f:
            json.dump(mixed, f, ensure_ascii=False)
        nb = len(out) * args.mix_repeat
        print(f"mix: {len(base)} base + {nb} bbox ({nb / len(mixed):.1%}) -> {mix_out}")


if __name__ == "__main__":
    main()
