#!/usr/bin/env python3
"""Record question-conditioned DAT sampling behaviour on HR-Bench 4K samples.

Motivation
----------
HR-Bench 4K stores 800 rows over only 177 distinct images, and 19 of those
images carry several *genuinely different* questions.  Those are exactly the
samples needed to show that DAT's deformable sampling is driven by the question
and not just by image saliency: same pixels, different question, different
sampling locations.

What is captured
----------------
Inference runs through the real lmms-eval wrapper (``Qwen3_5_DAT``), so the
geometry, prompt and answer match a normal eval run.  Two tensors are pulled out
of every DAT layer by monkey-patching (the in-model ``_want_vis`` path is gated
on ``self.training`` and therefore unreachable at eval time):

``guide``  [Lp, lr_h, lr_w]
    Intention->LR spatial attention, i.e. where the question token looks on the
    LR grid.  Recomputed here with the same formula the layer uses internally,
    since the model keeps it as a local.

``locs``   [Lp, off_grps, grid, grid, 2]
    Final sampling locations on the HD feature map, normalized to [-1, 1].
    Channel 0 is y, channel 1 is x -- ``grid_sample`` is fed ``locs[..., (1,0)]``
    with ``align_corners=True``.

Only the prefill call is kept; decode steps re-enter the layer with no image
segment and would otherwise append duplicates.

Usage
-----
    python scripts/vis_question_sampling.py \
        --ckpt /path/to/0901_sft_qwen35_4b_dat_ivcap-merged \
        --hashes 78363c71 7118c52a --out /path/to/vis_out
"""

import argparse
import base64
import glob
import hashlib
import io
import json
import math
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HRBENCH_GLOB = (
    "/home/ea-cv-nlp-train-offline-2/xzf/hf_eval/hub/"
    "datasets--DreamMr--HR-Bench/snapshots/*/hr_bench_4k.parquet"
)
# Matches the option-letter prompt lmms-eval uses for HR-Bench.
PROMPT_TMPL = (
    "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n"
    "Answer with the option's letter from the given choices directly."
)


def load_multi_question_images(parquet_glob, wanted=None):
    """Group HR-Bench rows by image, keeping images with >1 distinct question."""
    import pyarrow.parquet as pq

    path = glob.glob(parquet_glob)
    if not path:
        raise FileNotFoundError(f"no parquet matched {parquet_glob}")
    table = pq.read_table(path[0])
    col = {n: table.column(n).to_pylist() for n in table.column_names}

    by_image = defaultdict(list)
    for i, raw in enumerate(col["image"]):
        by_image[hashlib.md5(raw.encode()).hexdigest()[:8]].append(i)

    out = []
    for h, rows in by_image.items():
        if wanted and h not in wanted:
            continue
        first_row_of = {}
        for i in rows:
            first_row_of.setdefault(col["question"][i], i)
        if len(first_row_of) < 2:
            continue
        image = Image.open(io.BytesIO(base64.b64decode(col["image"][rows[0]]))).convert("RGB")
        out.append({
            "hash": h,
            "image": image,
            "questions": [
                {
                    "row": i,
                    "question": q,
                    "category": col["category"][i],
                    "gt": col["answer"][i],
                    "options": {k: col[k][i] for k in "ABCD"},
                }
                for q, i in first_row_of.items()
            ],
        })
    out.sort(key=lambda r: -len(r["questions"]))
    return out


def install_capture(dat_cls, mod):
    """Patch the sampling + attention entry points.

    Sampling locations alone do not settle whether DAT is question-conditioned:
    the points can stay put while the answer token redistributes its attention
    over them. So this also records

    ``attn``  [heads, n_ans, Ns] softmax(q @ k_hd^T) -- which sampled points the
              answer token actually reads. Recomputed from the same q/k the
              flash kernel gets, since the kernel never materializes it.

    ``w_hd``  [1, heads, n_ans] sigmoid(lse2 - lse1), the LSE-merge weight, i.e.
              how much of the merged output comes from the HD branch at all.

    Returns (store, reset, uninstall).
    """
    store = {"guide": defaultdict(list), "locs": defaultdict(list),
             "attn": [], "w_hd": defaultdict(list)}
    orig_gen = dat_cls._generate_offsets_and_sample
    orig_sample = dat_cls._sample_hd_from_off_guide
    orig_cross = mod._dat_cross_attn_varlen
    orig_merge = dat_cls._merge_two_pass_lse

    def patched_cross(q_list, k_list, v_list):
        for q, k in zip(q_list, k_list):
            d = q.shape[-1]
            logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * (d ** -0.5)
            store["attn"].append(logits.softmax(dim=-1).detach().cpu().numpy())
        return orig_cross(q_list, k_list, v_list)

    def patched_merge(self, out1, lse1, out2, lse2, ans_start, ans_end):
        l1 = lse1[:, :, ans_start:ans_end].float()
        store["w_hd"][self.layer_idx].append(
            torch.sigmoid(lse2.float() - l1).detach().cpu().numpy())
        return orig_merge(self, out1, lse1, out2, lse2, ans_start, ans_end)

    def patched_gen(self, query_states, image_hd_features, image_range_list,
                    b_idx, hd_feat_idxs, want_image=False):
        if self.use_intention_branch and getattr(self, "use_spatial_attn_guide", False):
            answer_ranges = image_range_list[b_idx][1:]
            intention_idx = [ar[2] for ar in answer_ranges]
            for (lr_start, lr_end, lr_h, lr_w) in image_range_list[b_idx][0]:
                idx = torch.arange(lr_start, lr_end, device=query_states.device)
                q_lr = query_states[b_idx, idx].float()
                q_int = query_states[b_idx, intention_idx].float()
                attn = torch.matmul(q_int, q_lr.transpose(0, 1)) / math.sqrt(q_lr.shape[-1])
                attn = attn.softmax(dim=-1).view(len(intention_idx), lr_h, lr_w)
                store["guide"][self.layer_idx].append(attn.detach().cpu().numpy())
        return orig_gen(self, query_states, image_hd_features, image_range_list,
                        b_idx, hd_feat_idxs, want_image=want_image)

    def patched_sample(self, off_guide, image_hd_features, hd_feat_idx, Lp, device, **kw):
        key, value, locs = orig_sample(self, off_guide, image_hd_features,
                                       hd_feat_idx, Lp, device, **kw)
        store["locs"][self.layer_idx].append(locs.detach().float().cpu().numpy())
        return key, value, locs

    dat_cls._generate_offsets_and_sample = patched_gen
    dat_cls._sample_hd_from_off_guide = patched_sample
    mod._dat_cross_attn_varlen = patched_cross
    dat_cls._merge_two_pass_lse = patched_merge

    def reset():
        store["guide"].clear()
        store["locs"].clear()
        store["w_hd"].clear()
        store["attn"].clear()

    def uninstall():
        dat_cls._generate_offsets_and_sample = orig_gen
        dat_cls._sample_hd_from_off_guide = orig_sample
        mod._dat_cross_attn_varlen = orig_cross
        dat_cls._merge_two_pass_lse = orig_merge

    return store, reset, uninstall


def build_wrapper(args):
    from lmms_eval.models.simple.qwen3_5_dat import Qwen3_5_DAT

    return Qwen3_5_DAT(
        pretrained=args.ckpt,
        attn_implementation="sdpa",
        hr_scale=args.hr_scale,
        max_pixels=args.hr_cap,
        min_pixels=args.min_pixels,
        lr_max_pixels=args.lr_pixels,
        lr_min_pixels=args.min_pixels,
        hd_early_exit_k=args.early_k,
        batch_size=1,
    )


def run_one(model, image, prompt, doc_id):
    """Single-request generate_until through the real eval path."""
    task, split = "vis", "test"
    model.task_dict = {task: {split: {doc_id: {"image": image}}}}
    req = SimpleNamespace(args=(
        prompt,
        {"max_new_tokens": 16, "temperature": 0.0, "do_sample": False},
        lambda doc: [doc["image"]],
        doc_id,
        task,
        split,
    ))
    return model.generate_until([req])[0]


def summarize(locs_by_layer):
    """Stack per-layer prefill sampling locations -> [n_layer, n_point, 2] (y, x)."""
    layers = sorted(locs_by_layer)
    pts = []
    for li in layers:
        first = locs_by_layer[li][0]        # prefill call only
        pts.append(first.reshape(-1, 2))    # (Lp*grps*grid*grid, 2)
    return np.stack(pts), layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hashes", nargs="*", default=None,
                    help="image hashes to run; default = all multi-question images")
    ap.add_argument("--max-images", type=int, default=4)
    ap.add_argument("--lr-pixels", type=int, default=1806336,
                    help="LR budget; 1806336 = 1344^2 = 1764 LLM visual tokens")
    ap.add_argument("--hr-cap", type=int, default=16257024,
                    help="HD cap; 16257024 = 4032^2 = HR-Bench 4K native")
    ap.add_argument("--min-pixels", type=int, default=28224)
    ap.add_argument("--hr-scale", type=int, default=3)
    ap.add_argument("--early-k", type=int, default=0)
    ap.add_argument("--parquet", default=HRBENCH_GLOB)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wanted = set(args.hashes) if args.hashes else None
    samples = load_multi_question_images(args.parquet, wanted)[: args.max_images]
    print(f"[vis] {len(samples)} image(s): "
          + ", ".join(f"{s['hash']}({len(s['questions'])}Q)" for s in samples), flush=True)

    model = build_wrapper(args)
    import llava.model.language_model.modeling_qwen3_5_dat as M
    store, reset, uninstall = install_capture(M.Qwen3_5AttentionDAT, M)

    manifest = []
    try:
        for s in samples:
            sub = os.path.join(args.out, s["hash"])
            os.makedirs(sub, exist_ok=True)
            s["image"].save(os.path.join(sub, "image.png"))
            rec = {"hash": s["hash"], "size": list(s["image"].size), "questions": []}

            for qi, q in enumerate(s["questions"]):
                reset()
                prompt = PROMPT_TMPL.format(question=q["question"], **q["options"])
                answer = run_one(model, s["image"], prompt, qi)

                if not store["locs"]:
                    print(f"[vis] !! no sampling captured for {s['hash']} q{qi}", flush=True)
                    continue
                locs, layers = summarize(store["locs"])
                nl = len(layers)
                guide = np.stack([store["guide"][li][0] for li in layers]) \
                    if store["guide"] else None
                # prefill only: decode steps re-enter and append more
                attn = np.stack(store["attn"][:nl]) if len(store["attn"]) >= nl else None
                w_hd = np.stack([store["w_hd"][li][0] for li in layers]) \
                    if store["w_hd"] else None

                np.savez_compressed(
                    os.path.join(sub, f"q{qi}.npz"),
                    locs=locs.astype(np.float32),
                    layers=np.array(layers),
                    **({"guide": guide.astype(np.float32)} if guide is not None else {}),
                    **({"attn": attn.astype(np.float32)} if attn is not None else {}),
                    **({"w_hd": w_hd.astype(np.float32)} if w_hd is not None else {}),
                )
                rec["questions"].append({
                    "idx": qi, "question": q["question"], "category": q["category"],
                    "gt": q["gt"], "options": q["options"],
                    "pred": answer.strip(), "correct": answer.strip().upper().startswith(q["gt"]),
                    "n_layer": nl, "n_point": int(locs.shape[1]),
                    "guide_shape": list(guide.shape) if guide is not None else None,
                    "attn_shape": list(attn.shape) if attn is not None else None,
                    "w_hd_mean": float(w_hd.mean()) if w_hd is not None else None,
                })
                extra = ""
                if w_hd is not None:
                    extra += "  w_hd=%.4f" % w_hd.mean()
                if attn is not None:
                    # peak/uniform tells how concentrated the read is over Ns points
                    ns = attn.shape[-1]
                    extra += "  attn_peak/unif=%.1f" % (attn.max(-1).mean() * ns)
                print(f"[vis] {s['hash']} q{qi} [{q['category']}] "
                      f"pred={answer.strip()[:20]!r} gt={q['gt']} "
                      f"layers={nl} pts/layer={locs.shape[1]}{extra}", flush=True)

            manifest.append(rec)
            json.dump(rec, open(os.path.join(sub, "meta.json"), "w"), indent=1)
    finally:
        uninstall()

    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"[vis] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
