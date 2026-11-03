"""Main-table rows from patch_sample_probe.py outputs: accuracy, mean prompt tokens, s/q.

python scripts/main_table_summary.py --main $X --mmdat $E/loc/psp_model
"""
import argparse
import glob
import json
import os
import re

DS = (("hrbench4k", "hr4k"), ("hrbench8k", "hr8k"), ("vstar", "vstar"))


def collect(jsons, logs):
    ok = n = tok = 0
    for f in jsons:
        for rows in json.load(open(f)).values():
            ok += sum(r["ok"] for r in rows)
            n += len(rows)
            tok += sum(r.get("n_tokens") or 0 for r in rows)
    sq = []
    for f in logs:
        for m in re.finditer(r"\[psp\] \S+: (\d+)/(\d+) \S+ ([\d.]+) s/q", open(f, errors="ignore").read()):
            sq.append((int(m.group(2)), float(m.group(3))))
    lat = sum(k * v for k, v in sq) / max(sum(k for k, _ in sq), 1) if sq else None
    return ok, n, tok / max(n, 1), lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="dir with {native_lr,native_hr,sft_lr}_{dataset}_{i}.json")
    ap.add_argument("--logs", default="/tmp/main_table")
    ap.add_argument("--mmdat", default="", help="dir with vstar*.json, hr4k_*.json, hr8k_*.json (+ .log)")
    args = ap.parse_args()
    rows = []
    for tag, name in (("native_lr", "Native (LR)"), ("native_hr", "Native (HR)"), ("sft_lr", "SFT control (LR)")):
        rows.append((name, [collect(glob.glob(f"{os.path.expanduser(args.main)}/{tag}_{d}_*.json"),
                                    glob.glob(f"{args.logs}/{tag}_{d}_*.log")) for d, _ in DS]))
    if args.mmdat:
        m = os.path.expanduser(args.mmdat)
        rows.append(("MM-DAT (ours)", [collect(glob.glob(f"{m}/{s}*.json"), glob.glob(f"{m}/{s}*.log")) for _, s in DS]))
    print(f"{'row':18s} " + "  ".join(f"{d:>22s}" for d, _ in DS) + "   avg3   tok(k)  s/q")
    for name, res in rows:
        acc = [100 * ok / n if n else None for ok, n, _, _ in res]
        cells = "  ".join(f"{ok:4d}/{n:<4d} {a:6.2f}%      " if a is not None else f"{'TBD':>22s}"
                          for (ok, n, _, _), a in zip(res, acc))
        avg = f"{sum(acc) / 3:6.2f}" if all(a is not None for a in acc) else "   TBD"
        toks = [t for ok, n, t, _ in res if n]
        lats = [l for *_, l in res if l is not None]
        tk = f"{sum(toks) / len(toks) / 1000:6.2f}" if toks else "   TBD"
        lt = f"{sum(lats) / len(lats):5.2f}" if lats else "  TBD"
        print(f"{name:18s} {cells}  {avg}  {tk}  {lt}")
    print("tok = mean prompt tokens per question (LR image + crop/DAT tokens + text), averaged over the 3 sets; "
          "s/q = per-question wall time on one GPU incl. generation")


if __name__ == "__main__":
    main()
