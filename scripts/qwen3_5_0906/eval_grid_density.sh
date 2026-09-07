#!/usr/bin/env bash
set -uo pipefail

# Grid-density sweep, INFERENCE ONLY, on the EE6 checkpoint at LR 1344² / HR 4032².
#
# Question: is the HR branch worth only +2-3 pp because 400 sampled positions
# (a 20x20 grid over a 126x126 HD map, pitch ~200 px on the original image) are
# far too sparse to land on the small targets HR-Bench asks about? If so, FSP
# should climb as the grid gets denser; if the branch only carries a global
# "second view", it stays flat.
#
# grid_size lives in config.dat_extra_args; conv_off_proj is 1x1 and the sampler
# works in normalized coordinates, so any grid runs on the trained weights.
#
#   uncorrected  g20 (control) g40 g60 g126  -> 400 / 1,600 / 3,600 / 15,876 positions
#                 (x8 offset groups = keys per DAT layer: 3.2k / 12.8k / 28.8k / 127k)
#   corrected    g40c g60c g126c: hd_lse_bias = -log(N/400) so the LSE merge weight
#                 w_hd = sigmoid(lse2 - lse1) is not inflated by the larger key set
#                 (4x keys raise lse2 by ~1.39, which alone would move w_hd 0.2 -> 0.5).
#
# Then latency for the four uncorrected grids at the same operating point.
#
# Usage (tmux on the idle eval pod):
#   bash scripts/qwen3_5_0906/eval_grid_density.sh

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source scripts/eval_pod_env.sh
export PYTHONUNBUFFERED=1

X="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
SRC="${SRC:-$X/vldat_experiments/0901_sft_qwen35_4b_dat_ivcap_ee6-merged}"
WORK="$X/vldat_experiments/_grid_arms"
LOG="$X/vldat_experiments/logs_0907_grid"
PIXELS="${PIXELS:-1806336}"          # LR 1344² -> 1764 tokens, HR 4032²
export HR_CAP="${HR_CAP:-16257024}"
export HR_SCALE=3
mkdir -p "$WORK" "$LOG"

for p in $(pgrep -f "auto_burn.py" 2>/dev/null || true); do pkill -P "$p" 2>/dev/null || true; done
sleep 5

python3 - "$SRC" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))["dat_extra_args"]
print("[grid] source ckpt dat_extra_args: grid_size=%s hd_early_exit_k=%s off_grps=%s" % (
    c.get("grid_size"), c.get("hd_early_exit_k"), c.get("off_grps")))
PY

make_arm() { # name grid bias
  local arm="$WORK/$1"
  rm -rf "$arm"; mkdir -p "$arm"
  for f in "$SRC"/*; do ln -s "$f" "$arm/"; done
  rm -f "$arm/config.json"
  python3 - "$SRC" "$arm" "$2" "$3" <<'PY'
import json, sys
src, dst, g, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
c = json.load(open(f"{src}/config.json"))
c["dat_extra_args"]["grid_size"] = g
c["dat_extra_args"]["hd_lse_bias"] = b
json.dump(c, open(f"{dst}/config.json", "w"), indent=1)
print(f"[grid] arm {dst.split('/')[-1]}: grid_size={g} ({g*g} positions) hd_lse_bias={b:+.4f}")
PY
}

# name grid bias   (bias = -ln(N/400))
ARMS=(
  "g20   20  0"
  "g40   40  0"
  "g60   60  0"
  "g126 126  0"
  "g40c  40 -1.3863"
  "g60c  60 -2.1972"
  "g126c 126 -3.6810"
)

STAMP="$(date +%m%d_%H%M%S)"
i=0
for spec in "${ARMS[@]}"; do
  read -r name grid bias <<< "$spec"
  make_arm "$name" "$grid" "$bias"
  export PORT=$((37000 + i * 100)); i=$((i + 1))
  echo
  echo "########## [$(date '+%F %T')] $name  grid=$grid  bias=$bias  PIXELS=$PIXELS  port=$PORT ##########"
  if ! { PIXELS="$PIXELS" bash scripts/eval_pixel_sweep.sh dat35 "$WORK/$name" hrbench4k "0907grid_$name" \
        2>&1 | tee "$LOG/${STAMP}_hrbench4k_$name.log" | grep -E "^\[dat35|Traceback|Error|OutOfMemory"; }; then
    echo "[grid] $name FAILED — continuing" >&2
  fi
done

echo
echo "########## [$(date '+%F %T')] latency at tok1764 for the uncorrected grids ##########"
source "$VLDAT_VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=0
BASE_MODEL=/workspace/model_cache/Qwen3.5-4B
[[ -f "$BASE_MODEL/config.json" ]] || BASE_MODEL="$X/models/Qwen3.5-4B"
for name in g20 g40 g60 g126; do
  echo "--- [$(date '+%T')] pareto $name ---"
  python test_inference_bench.py \
      --model-family qwen3_5 --base-model "$BASE_MODEL" --dat-ckpt "$WORK/$name" \
      --synthetic --tasks pareto --pareto-tokens 1764 --hr-cap "$HR_CAP" --hd-early-k 6 \
      --no-native --warmup 2 --iters 5 --tag "pareto_grid_${name}_${STAMP}" 2>&1 \
      | grep -E "tok=|qwen=|dat=|Done|Error|OutOfMemory" || echo "[grid] pareto $name FAILED" >&2
done

echo
echo "########## [$(date '+%F %T')] GRID SWEEP DONE ##########"
