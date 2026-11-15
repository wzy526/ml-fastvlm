#!/usr/bin/env python
"""Score saved V* generations with both lmms-eval scorers.

lmms-eval changed two things at once: the prompt (e29760d9 / eaed5534 stop
appending the answer instruction a second time) and the scorer (the lenient
first-letter regex became a strict last-line reader). Re-scoring the same
responses with both scorers separates the two.

    python scripts/vstar_rescore.py DIR [DIR ...] [--pair OLD NEW] [--show 8]

Each DIR is searched for *vstar_bench*.jsonl sample files.
"""
import argparse
import glob
import json
import os
import re
import time
from collections import Counter

INSTRUCTION = "Answer with the option's letter from the given choices directly."


def old_letter(response):
    response = str(response).strip().upper()
    patterns = [
        r"^([A-D])\s*[\.)\]]*",
        r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+([A-D])",
        r"\(([A-D])\)",
        r"([A-D])\s*(?:\.|\)|])",
        r"(?:^|\s)([A-D])(?:\s|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    letters = re.findall(r"[A-D]", response)
    if len(letters) == 1:
        return letters[0]
    if response and response[0] in "ABCD":
        return response[0]
    return None


def new_letter(response):
    text = str(response).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    elif "<think>" in text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = re.fullmatch(r"[\s*`(\[]*([A-D])[\s*`)\].!]*", lines[-1], re.IGNORECASE)
        if last:
            return last.group(1).upper()
        option_only = re.fullmatch(
            r"(?:the\s+)?(?:(?:correct|final|best)\s+)?option\s*(?:is\b|:|=)?"
            r"[\s*`(\[]*([A-D])[\s*`)\].!]*",
            lines[-1],
            re.IGNORECASE,
        )
        if option_only:
            return option_only.group(1).upper()
    stated = re.findall(
        r"\b(?:answer|choice)\b[\s\"'*`]*(?:is\b|:|：|=)?"
        r"[\s\"'*`(]*([A-D])\b|\\boxed\{\s*([A-D])\s*\}",
        text,
        re.IGNORECASE,
    )
    if stated:
        return next(letter for letter in stated[-1] if letter).upper()
    if lines:
        option = re.match(r"^[*`]*\(?([A-D])(?:[.)\]]|\s*[-:])\s+", lines[-1], re.IGNORECASE)
        if option:
            return option.group(1).upper()
    return None


def _text(value):
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return "" if value is None or isinstance(value, (list, tuple)) else str(value)


def _form(response):
    r = response.strip()
    if re.fullmatch(r"[\s*`(\[]*[A-D][\s*`)\].!]*", r):
        return "letter"
    if re.match(r"^[*`]*\(?[A-D](?:[.)\]]|\s*[-:])", r):
        return "letter+text"
    if not r:
        return "empty"
    return "prose"


def load(path):
    rows = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        meta = r.get("vstar_overall_acc") or {}
        gt = meta.get("ground_truth") if isinstance(meta, dict) else None
        gt = (gt or _text(r.get("target"))).strip().upper()
        resp = _text(r.get("filtered_resps", r.get("resps")))
        prompt = r.get("input")
        if prompt is None:
            prompt = _text(r.get("arguments"))
        cat = meta.get("category", "?") if isinstance(meta, dict) else "?"
        rows[str(r["doc_id"])] = dict(gt=gt, resp=resp, prompt=str(prompt), cat=cat)
    return rows


def find(d):
    return sorted(glob.glob(os.path.join(d, "**", "*vstar_bench*.jsonl"), recursive=True))


def summary(path, rows):
    n = len(rows)
    old = sum(old_letter(r["resp"]) == r["gt"] for r in rows.values())
    new = sum(new_letter(r["resp"]) == r["gt"] for r in rows.values())
    unparsed = sum(new_letter(r["resp"]) is None for r in rows.values())
    inst = Counter(r["prompt"].count(INSTRUCTION) for r in rows.values())
    forms = Counter(_form(r["resp"]) for r in rows.values())
    when = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
    print(f"{path}\n    {when}  n={n}  指令次数 {dict(inst)}  旧打分 {old}/{n}  新打分 {new}/{n}  "
          f"新打分解析失败 {unparsed}  回答形式 {dict(forms)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--pair", nargs=2, metavar=("OLD", "NEW"), help="two sample dirs to compare per question")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    for d in args.dirs:
        files = find(d)
        if not files:
            print(f"{d}: 无 vstar samples")
        for f in files:
            summary(f, load(f))

    if args.pair:
        fa, fb = find(args.pair[0]), find(args.pair[1])
        if not fa or not fb:
            print("pair: 缺 samples")
            return
        a, b = load(fa[-1]), load(fb[-1])
        ids = sorted(set(a) & set(b), key=int)
        same_resp = sum(a[i]["resp"].strip() == b[i]["resp"].strip() for i in ids)
        la = {i: old_letter(a[i]["resp"]) for i in ids}
        lb = {i: old_letter(b[i]["resp"]) for i in ids}
        same_letter = sum(la[i] == lb[i] for i in ids)
        gain = [i for i in ids if la[i] != a[i]["gt"] and lb[i] == b[i]["gt"]]
        loss = [i for i in ids if la[i] == a[i]["gt"] and lb[i] != b[i]["gt"]]
        print(f"\npair {len(ids)} 题: 回答逐字相同 {same_resp}  旧打分字母相同 {same_letter}  "
              f"同一打分(旧)下 NEW 多对 {len(gain)} 少对 {len(loss)}")
        cats = Counter(a[i]["cat"] for i in loss)
        print(f"    少对的类别 {dict(cats)}")
        for i in loss[: args.show]:
            print(f"    doc {i} gt {a[i]['gt']}  OLD {a[i]['resp']!r}  NEW {b[i]['resp']!r}")
        for i in gain[: max(0, args.show // 2)]:
            print(f"    +doc {i} gt {a[i]['gt']}  OLD {a[i]['resp']!r}  NEW {b[i]['resp']!r}")


if __name__ == "__main__":
    main()
