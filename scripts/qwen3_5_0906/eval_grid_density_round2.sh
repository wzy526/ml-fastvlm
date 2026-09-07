#!/usr/bin/env bash
set -uo pipefail
# Round 2 of the grid-density sweep (see eval_grid_density.sh).
#
# Round 1 showed: denser grids with the merge weight left free lose accuracy
# (w_hd 0.14 -> 0.26 / 0.35 / 0.54 as N grows), while -log(N/400) over-corrects
# (effective w_hd 0.08 / 0.06 / 0.03) and lands within noise of g20. Two things
# are still open:
#   * equal-w_hd arms: bias = -(logit(w_hd_N) - logit(w_hd_20)) from the measured
#     vis values, so every grid merges at g20's 0.14 -> isolates density itself.
#   * what the HD branch is worth at all: bias -30 (w_hd ~ 0) removes it, so the
#     remaining gap to vanilla is the LoRA-SFT'd LLM, not HD; bias +0.76 doubles
#     w_hd at grid 20, telling whether round-1 losses were weight or extra keys.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source scripts/eval_pod_env.sh
export PYTHONUNBUFFERED=1
X="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
SRC="$X/vldat_experiments/0901_sft_qwen35_4b_dat_ivcap_ee6-merged"
WORK="$X/vldat_experiments/_grid_arms"
LOG="$X/vldat_experiments/logs_0907_grid"
PIXELS=1806336
export HR_CAP=16257024 HR_SCALE=3
for p in $(pgrep -f "auto_burn.py" 2>/dev/null || true); do pkill -P "$p" 2>/dev/null || true; done
sleep 5
make_arm() {
  local arm="$WORK/$1"; rm -rf "$arm"; mkdir -p "$arm"
  for f in "$SRC"/*; do ln -s "$f" "$arm/"; done
  rm -f "$arm/config.json"
  python3 - "$SRC" "$arm" "$2" "$3" <<'PY'
import json, sys
src, dst, g, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
c = json.load(open(f"{src}/config.json"))
c["dat_extra_args"]["grid_size"] = g
c["dat_extra_args"]["hd_lse_bias"] = b
json.dump(c, open(f"{dst}/config.json", "w"), indent=1)
print(f"[grid2] arm {dst.split('/')[-1]}: grid_size={g} hd_lse_bias={b:+.3f}")
PY
}
ARMS=(
  "g20_nohd  20 -30"
  "g20_up    20  0.76"
  "g40e      40 -0.76"
  "g60e      60 -1.21"
  "g126e    126 -1.99"
)
STAMP="$(date +%m%d_%H%M%S)"; i=0
for spec in "${ARMS[@]}"; do
  read -r name grid bias <<< "$spec"
  make_arm "$name" "$grid" "$bias"
  export PORT=$((38000 + i * 100)); i=$((i + 1))
  echo; echo "########## [$(date '+%F %T')] $name grid=$grid bias=$bias port=$PORT ##########"
  if ! { PIXELS="$PIXELS" bash scripts/eval_pixel_sweep.sh dat35 "$WORK/$name" hrbench4k "0907grid_$name" \
        2>&1 | tee "$LOG/${STAMP}_hrbench4k_$name.log" | grep -E "^\[dat35|Traceback|Error|OutOfMemory"; }; then
    echo "[grid2] $name FAILED — continuing" >&2
  fi
done
echo "########## [$(date '+%F %T')] ROUND 2 DONE ##########"
