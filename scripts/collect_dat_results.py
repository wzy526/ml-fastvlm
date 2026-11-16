#!/usr/bin/env python
"""One table of every lmms-eval result under the given directories.

Rows are runs (an eval output dir and its LR budget), columns are tasks with
one headline metric each; values <= 1 are shown as percentages. The legend
under the table gives each row's weights and switches, and for V* whether the
answer instruction appeared once (new prompt) or twice (old prompt).

    python scripts/collect_dat_results.py DIR [DIR ...] [--all] [--detail] [--csv out.csv]

By default only the DAT adapters (qwen3_5_ldat / qwen2_5_ldat) are listed;
--all also lists native models.
"""
import argparse
import csv
import glob
import json
import os
import re
import time

INSTRUCTION = "Answer with the option's letter from the given choices directly."

COLUMNS = [
    ("vstar_bench", "V*", ["vstar_overall_acc"]),
    ("hrbench4k", "HR4K", ["average"]),
    ("hrbench8k", "HR8K", ["average"]),
    ("chartqa", "Chart", ["relaxed_overall"]),
    ("docvqa_val", "DocVQA", ["anls"]),
    ("textvqa_val", "TextVQA", ["exact_match"]),
    ("refcoco_bbox_rec_val", "RefCOCO", ["ACC@0.5", "refcoco_ACC@0.5", "IoU"]),
    ("gqa", "GQA", ["exact_match"]),
    ("scienceqa_img", "SQA", ["exact_match"]),
    ("vizwiz_vqa_val", "VizWiz", ["exact_match"]),
    ("pope", "POPE", ["pope_accuracy", "pope_f1_score"]),
    ("mmbench_en_dev", "MMB", ["gpt_eval_score", "mmbench_en_dev_score", "submission"]),
    ("seedbench", "SEED", ["seed_all", "seed_image"]),
    ("mme", "MME", ["mme_perception_score"]),
]
ALIASES = {"hrbench4k_local": "hrbench4k", "hrbench8k_local": "hrbench8k"}
DETAIL_EXTRA = {
    "vstar_bench": ["vstar_direct_attributes_acc", "vstar_relative_position_acc"],
    "hrbench4k": ["single", "cross"],
    "hrbench8k": ["single", "cross"],
    "mme": ["mme_cognition_score"],
}


def metrics(block):
    out = {}
    for k, v in block.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or "stderr" in k:
            continue
        out[k.split(",")[0]] = float(v)
    return out


def shown(v):
    return v * 100 if abs(v) <= 1.0 else v


def parse_args_string(s):
    out = {}
    for item in filter(None, (s or "").split(",")):
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out


def run_key(path, base):
    rel = os.path.relpath(path, base)
    parts = rel.split(os.sep)
    for i, p in enumerate(parts):
        m = re.fullmatch(r"(.+)_lr(\d+)", p)
        if m:
            return os.path.join(os.path.basename(base), *parts[:i]), m.group(2)
    if "results" in parts:
        i = parts.index("results")
        return os.path.join(os.path.basename(base), *parts[:i]), ""
    return os.path.join(os.path.basename(base), *parts[:-2]), ""


def vstar_prompt(results_path):
    d = os.path.dirname(results_path)
    stamp = os.path.basename(results_path).split("_results")[0]
    files = sorted(glob.glob(os.path.join(d, f"{stamp}*vstar_bench*.jsonl"))) or \
        sorted(glob.glob(os.path.join(d, "*vstar_bench*.jsonl")))
    if not files:
        return ""
    counts = set()
    with open(files[-1]) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                p = r.get("input")
                if p is None:
                    a = r.get("arguments") or [""]
                    while isinstance(a, (list, tuple)) and a:
                        a = a[0]
                    p = a
                counts.add(str(p).count(INSTRUCTION))
    if counts == {1}:
        return "新prompt"
    if counts == {2}:
        return "旧prompt"
    return f"prompt{sorted(counts)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--all", action="store_true", help="also list native (non-DAT) models")
    ap.add_argument("--detail", action="store_true", help="print every metric of every row")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    runs = {}
    for base in args.dirs:
        base = os.path.abspath(os.path.expanduser(base))
        for path in glob.glob(os.path.join(base, "**", "*results.json"), recursive=True):
            try:
                data = json.load(open(path))
            except (OSError, ValueError):
                continue
            if "results" not in data:
                continue
            cfg = data.get("config", {})
            model = str(cfg.get("model", ""))
            if not args.all and "ldat" not in model:
                continue
            ma = parse_args_string(cfg.get("model_args", ""))
            key = run_key(path, base)
            lr = ma.get("lr_tok") or key[1]
            row = runs.setdefault((key[0], lr), {"tasks": {}, "model": model, "args": ma, "paths": {}})
            mtime = os.path.getmtime(path)
            for task, block in data["results"].items():
                task = ALIASES.get(task, task)
                prev = row["tasks"].get(task)
                if prev is None or prev["mtime"] < mtime:
                    extra = vstar_prompt(path) if task == "vstar_bench" else ""
                    row["tasks"][task] = {"m": metrics(block), "mtime": mtime, "prompt": extra}
                    row["paths"][task] = path
                    row["args"] = ma or row["args"]

    if not runs:
        print("没有找到结果")
        return

    order = sorted(runs, key=lambda k: (min(t["mtime"] for t in runs[k]["tasks"].values()), k))
    head = f"{'id':<4}" + "".join(f"{c[1]:>8}" for c in COLUMNS)
    print(head)
    table = []
    for i, k in enumerate(order, 1):
        row = runs[k]
        cells, rec = [], {"id": f"R{i}", "run": k[0], "lr_tok": k[1]}
        for task, short, prefs in COLUMNS:
            t = row["tasks"].get(task)
            val = None
            if t:
                for p in prefs:
                    if p in t["m"]:
                        val = shown(t["m"][p])
                        break
                if val is None and t["m"]:
                    val = shown(next(iter(t["m"].values())))
            cells.append(f"{val:8.2f}" if val is not None else f"{'-':>8}")
            rec[short] = round(val, 2) if val is not None else ""
        print(f"R{i:<3}" + "".join(cells))
        table.append(rec)

    print("\n图例")
    for i, k in enumerate(order, 1):
        row = runs[k]
        ma = row["args"]
        flags = [f"{n}={ma[n]}" for n in ("crop_tok", "n_win", "grid", "dat_alpha", "dat_k", "win_full",
                                           "where_layer", "fast_prefix", "sample_feat",
                                           "prefix_cache", "enable_thinking") if n in ma]
        first = min(t["mtime"] for t in row["tasks"].values())
        vs = row["tasks"].get("vstar_bench", {}).get("prompt", "")
        print(f"R{i:<3} {time.strftime('%m-%d %H:%M', time.localtime(first))}  {k[0]}  lr={k[1] or '?'}  "
              f"{row['model']}")
        print(f"     model={os.path.basename(ma.get('pretrained', '').rstrip('/'))}  "
              f"ldat={os.path.basename(ma.get('ldat_pt', ''))}  " + " ".join(flags) +
              (f"  V*:{vs}" if vs else ""))
        if args.detail:
            for task, _, _ in COLUMNS:
                t = row["tasks"].get(task)
                if t:
                    keep = {m: round(shown(v), 2) for m, v in t["m"].items()}
                    print(f"       {task}: {keep}")
            for task in sorted(set(row["tasks"]) - {c[0] for c in COLUMNS}):
                keep = {m: round(shown(v), 2) for m, v in row["tasks"][task]["m"].items()}
                print(f"       {task}: {keep}")
        else:
            for task, names in DETAIL_EXTRA.items():
                t = row["tasks"].get(task)
                if t:
                    parts = [f"{n.replace('vstar_', '').replace('_acc', '')} {shown(t['m'][n]):.2f}"
                             for n in names if n in t["m"]]
                    if parts:
                        print(f"       {task}: " + "  ".join(parts))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f"\ncsv: {args.csv}")


if __name__ == "__main__":
    main()
