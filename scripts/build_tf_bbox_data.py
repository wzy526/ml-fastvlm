#!/usr/bin/env python3
"""Build a teacher-forcing SFT json: explode "box-in-question" turns into
single-turn samples carrying a top-level ``bbox`` field.

Why: the V* oracle probe showed that forcing the HD grid onto the GT box does
not help a model whose readout (K/V_hd) never learned to use localized detail.
Teacher forcing during SFT gives the readout detail at the right place; the
window is read by ``Qwen2VLCoupledDATDataset._teacher_force_window`` from the
``bbox`` field (fractions of the image, [x0, y0, x1, y1]).

Which turns qualify: a (human, gpt) pair where the human turn contains a
normalized box "[x0, y0, x1, y1]" and the gpt turn contains none — e.g. VG
"Please provide a short description for this region: [0.0, 0.64, 0.3, 0.76]".
Grounding turns (box in the answer) are NOT used: the box is the answer there,
so forcing would leak it.

Output = every original sample unchanged (unless --drop_originals) + up to
--per_image exploded single-turn samples per source sample, each with ``bbox``.

Usage:
  python scripts/build_tf_bbox_data.py \
      --src /data/oss_bucket_0/wangziyi/models_data/llava_hr_gen_vs_0817.json \
      --dst /data/oss_bucket_0/wangziyi/models_data/llava_hr_gen_vs_0817_tfbox.json \
      --per_image 2 --seed 0
"""
import argparse
import collections
import json
import random
import re

BOX = re.compile(r"\[\s*(\d(?:\.\d+)?)\s*,\s*(\d(?:\.\d+)?)\s*,\s*(\d(?:\.\d+)?)\s*,\s*(\d(?:\.\d+)?)\s*\]")


def boxes_in(text):
    out = []
    for m in BOX.finditer(text or ""):
        b = [float(v) for v in m.groups()]
        if all(0.0 <= v <= 1.0 for v in b) and b[2] > b[0] and b[3] > b[1]:
            out.append(b)
    return out


def source_of(x):
    s = str(x.get("id", "?")).split("_")[0][:20]
    return s if not s.isdigit() else "numeric"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--per_image", type=int, default=2,
                    help="max exploded box-in-Q singles per source sample")
    ap.add_argument("--min_side", type=float, default=0.0,
                    help="skip boxes whose shorter side (fraction) is below this")
    ap.add_argument("--max_side", type=float, default=0.6,
                    help="skip boxes whose longer side exceeds this (near-whole-image "
                         "regions need no HD detail)")
    ap.add_argument("--drop_originals", action="store_true",
                    help="drop the original multi-turn samples that were exploded")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    data = json.load(open(args.src))
    out, n_expl, n_src_with = [], collections.Counter(), collections.Counter()
    for x in data:
        conv = x.get("conversations", [])
        cands = []
        for i in range(len(conv) - 1):
            q, a = conv[i], conv[i + 1]
            if q.get("from") not in ("human", "user") or a.get("from") not in ("gpt", "assistant"):
                continue
            qb = boxes_in(q.get("value", ""))
            if len(qb) != 1 or boxes_in(a.get("value", "")):
                continue
            b = qb[0]
            w, h = b[2] - b[0], b[3] - b[1]
            if min(w, h) < args.min_side or max(w, h) > args.max_side:
                continue
            cands.append((i, b))
        src = source_of(x)
        keep_orig = not (args.drop_originals and cands)
        if keep_orig:
            out.append(x)
        if not cands:
            continue
        n_src_with[src] += 1
        random.shuffle(cands)
        for k, (i, b) in enumerate(cands[: args.per_image]):
            qv = conv[i]["value"]
            if "<image>" not in qv:
                qv = "<image>\n" + qv
            y = {kk: vv for kk, vv in x.items() if kk not in ("conversations", "index")}
            y["id"] = f"{x.get('id', 'x')}_tf{k}"
            y["conversations"] = [{"from": "human", "value": qv},
                                  {"from": "gpt", "value": conv[i + 1]["value"]}]
            y["bbox"] = [round(v, 4) for v in b]
            out.append(y)
            n_expl[src] += 1

    random.shuffle(out)
    json.dump(out, open(args.dst, "w"), ensure_ascii=False)
    print(f"src {len(data)} -> dst {len(out)}  (+{sum(n_expl.values())} exploded singles with bbox)")
    print(f"{'source':>14} {'samples_with_boxQ':>18} {'exploded':>9}")
    for s, n in n_src_with.most_common():
        print(f"{s:>14} {n:>18} {n_expl[s]:>9}")
    ex = next(y for y in out if "bbox" in y)
    print("\nexample:", json.dumps(ex, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
