#!/usr/bin/env bash
set -uo pipefail
# Companion to eval_grid_density.sh: record the LSE merge weight w_hd for each grid
# (the eval itself does not log it). Runs on one GPU alongside the eval; timing is
# irrelevant here. Arms are the ones eval_grid_density.sh creates.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source scripts/eval_pod_env.sh
source "$VLDAT_VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
X="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
WORK="$X/vldat_experiments/_grid_arms"
LOG="$X/vldat_experiments/logs_0907_grid"
mkdir -p "$LOG"
for name in g20 g40 g60 g126 g40c g60c g126c; do
  while [[ ! -f "$WORK/$name/config.json" ]]; do sleep 30; done   # created by the eval driver
  echo "########## [$(date '+%F %T')] vis $name ##########"
  python scripts/vis_question_sampling.py --ckpt "$WORK/$name" --out "$X/vis_samples/grid_$name" \
      --max-images 3 --lr-pixels 1806336 --hr-cap 16257024 > "$LOG/vis_$name.log" 2>&1 \
      || echo "[vis] $name FAILED" >&2
  python3 - "$LOG/vis_$name.log" "$name" <<'PY'
import re, sys
w = [float(x) for x in re.findall(r"w_hd=([0-9.]+)", open(sys.argv[1]).read())]
p = [float(x) for x in re.findall(r"attn_peak/unif=([0-9.]+)", open(sys.argv[1]).read())]
if w:
    print("[vis] %-6s w_hd mean=%.4f (n=%d)  attn_peak/unif mean=%.1f" % (sys.argv[2], sum(w)/len(w), len(w), sum(p)/max(len(p),1)))
else:
    print("[vis] %-6s no w_hd lines (see log)" % sys.argv[2])
PY
done
echo "########## [$(date '+%F %T')] VIS DONE ##########"
