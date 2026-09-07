#!/usr/bin/env bash
set -euo pipefail

# 0906 resolution sweep — open up DAT's HR branch to native 4K.
#
# WHY THIS EXISTS
# ---------------
# The 0903 sweep pinned DAT's HR branch at HR_CAP=5017600 (2240²), which is only
# 30.9% of HR-Bench 4K's native pixel count (median 16.26 MP, 68% of rows are
# exactly 4032²) and 13.0% of the 8K subset's (median 38.75 MP). Worse, since
#   HR = min(LR * hr_scale², HR_CAP)
# the cap binds from 640 LLM tokens onward, and by 6400 tokens the nominal 3x
# ratio had collapsed BELOW 1x (effective 0.88, then 0.65 at 11520) — the HR
# branch was downsampling relative to the LR branch and could not contribute any
# detail at all. DAT's core mechanism was never exercised in that sweep.
#
# A SECOND, WORSE PROBLEM (fixed 2026-09-06)
# ------------------------------------------
# `_derive_hr_size_from_lr_first` in lmms-eval's qwen2_5_dat_vl.py computed
#   lr_h_px = image_grid_thw[1] * self._factor
# but image_grid_thw counts UNMERGED patches, so the edge scale is patch_size,
# not _factor = patch_size * spatial_merge. That overestimated lr_pixels by 4x,
# making hd_target 36x the LR area instead of hr_scale^2 = 9x — which is the
# real reason every point pinned to the cap. Fixed to use _patch_size (the
# HR-anchored path 20 lines above already did it correctly, and the function's
# own docstring says `lr_pixels = lr_h * lr_w * 14^2`).
#
# Under the OLD cap the fix only moves the 256-token point (2240² -> 1536²);
# every larger point was capped either way. So the 0903 accuracy numbers remain
# valid for the regime they actually measured ("HR pinned at 2240²") — they just
# never measured the intended 3x geometry.
#
# THE GRID (verified against the real processor, hr_scale=3 exact everywhere)
# --------------------------------------------------------------------------
#   lr_max_pixels  LR res   LRtok  HR res   HR MP   ratio  vs 4K native
#   262144         512²      256   1536²     2.36   3.00   14.5%
#   655360         800²      625   2400²     5.76   3.00   35.4%  <- near training
#   1310720       1120²     1225   3360²    11.29   3.00   69.4%
#   1806336       1344²     1764   4032²    16.26   3.00    100%
#
# Training for reference: lr_max_pixels=501760 -> LR 704² (484 tok), HR 2112².
# So 625 sits closest to the training distribution; 1764 is 3.6x extrapolation.
#
# BASE IS MATCHED ON HR RESOLUTION, NOT ON LR TOKENS. What we want to compare is
# "same visual information, how many LLM tokens does it cost" — so base runs at
# exactly DAT's HR resolutions:
#   1536² = 2359296 px =  2304 tok   vs DAT's  256 tok
#   2400² = 5760000 px =  5625 tok   vs DAT's  625 tok
#   3360² = 11289600 px = 11025 tok  vs DAT's 1225 tok
#   4032² = 16257024 px = 15876 tok  vs DAT's 1764 tok
# Every pair is a clean 9x token ratio at identical pixel coverage — the first
# accuracy comparison here at genuinely equal visual information. Earlier
# "speedups" either differed in token budget (the same-resolution bench) or
# handicapped the HR branch (the 0903 grid).
#
# USAGE (on the GPU host, after EE6 merge):
#   source scripts/eval_pod_env.sh
#   TIER=1 bash scripts/qwen3_5_0906/eval_resolution_sweep.sh
#
#   TIER=1   ~7 points   k0 + ee12 on hrbench4k, plus base's native-4K point
#   TIER=2   +ee6/ee18 on hrbench4k  (needs the EE6 merge to exist)
#   TIER=3   +hrbench8k and vstar_bench for all arms
#
# DRY_RUN=1 prints the plan (and which checkpoints are missing) without touching
# a GPU — use it to validate the orchestration while training still occupies the
# card.
#
# Resume-safe: eval_pixel_sweep.sh skips points that already have results, so
# re-running after an interruption costs nothing.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TIER="${TIER:-1}"
TAG="${TAG:-0906_res}"

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
LOG_DIR="${LOG_DIR:-$XZF_ROOT/vldat_experiments/logs_0906_res}"
mkdir -p "$LOG_DIR"

# Native-4K HR cap. hr_scale stays at its trained value of 3.
export HR_CAP="${HR_CAP:-16257024}"
export HR_SCALE="${HR_SCALE:-3}"

# LR grid for DAT: every point keeps hr_scale=3 exact (no truncation).
DAT_PIXELS="${DAT_PIXELS:-262144 655360 1310720 1806336}"
# base matched to DAT's HR resolutions (1536² / 2400² / 3360² / 4032²), so each
# pair sees identical pixels at a 9x token ratio. None of these coincide with
# the 0903 base grid, so all four are new.
BASE_PIXELS="${BASE_PIXELS:-2359296 5760000 11289600 16257024}"

pick_base() {
    for p in /workspace/model_cache/Qwen3.5-4B "$XZF_ROOT/models/Qwen3.5-4B"; do
        [[ -f "$p/config.json" ]] && { echo "$p"; return; }
    done
    echo "[ERROR] Qwen3.5-4B base not found" >&2; exit 1
}
BASE_4B="$(pick_base)"

K0="$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap-merged"
EE12="$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap_ee12-merged"
EE18="$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap_ee18-merged"
EE6="$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap_ee6-merged"

# spec format: kind|ckpt|label|pixels|task
SPECS=()
add() { SPECS+=("$1|$2|$3|$4|$5"); }

# --- TIER 1: does accuracy track HR pixels? ---------------------------------
add dat35  "$K0"     4b_dat_k0   "$DAT_PIXELS"  hrbench4k
add dat35  "$EE12"   4b_dat_ee12 "$DAT_PIXELS"  hrbench4k
add base35 "$BASE_4B" 4b_base    "$BASE_PIXELS" hrbench4k

# --- TIER 2: the other two early-exit arms ----------------------------------
if (( TIER >= 2 )); then
    add dat35 "$EE18" 4b_dat_ee18 "$DAT_PIXELS" hrbench4k
    add dat35 "$EE6"  4b_dat_ee6  "$DAT_PIXELS" hrbench4k
fi

# --- TIER 3: 8K subset + V* -------------------------------------------------
# NOTE: 16.26 MP is still only 42% of the 8K subset's native 38.75 MP, so this
# tier understates DAT there. Raising HR_CAP further for 8K is a separate run.
if (( TIER >= 3 )); then
    for t in hrbench8k vstar_bench; do
        add dat35  "$K0"      4b_dat_k0   "$DAT_PIXELS"  "$t"
        add dat35  "$EE12"    4b_dat_ee12 "$DAT_PIXELS"  "$t"
        add dat35  "$EE18"    4b_dat_ee18 "$DAT_PIXELS"  "$t"
        add dat35  "$EE6"     4b_dat_ee6  "$DAT_PIXELS"  "$t"
        add base35 "$BASE_4B" 4b_base     "$BASE_PIXELS" "$t"
    done
fi

STAMP="$(date +%m%d_%H%M%S)"
echo "[0906-res] tier=$TIER  HR_CAP=$HR_CAP  hr_scale=$HR_SCALE  specs=${#SPECS[@]}"
echo "[0906-res] logs -> $LOG_DIR"

RUN_IDX=0
for spec in "${SPECS[@]}"; do
    IFS='|' read -r kind ckpt label pixels task <<< "$spec"

    if [[ ! -f "$ckpt/config.json" ]]; then
        echo "[0906-res] SKIP $label ($task): ckpt missing -> $ckpt"
        continue
    fi

    # Unique port per launch; a stale rank from a previous point must not clash.
    export PORT=$((33000 + RUN_IDX * 100))
    RUN_IDX=$((RUN_IDX + 1))

    LOG="$LOG_DIR/${STAMP}_${task}_${label}.log"
    echo
    echo "########## [$(date '+%F %T')] $label / $task  px=[$pixels]  port=$PORT ##########"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "  DRY_RUN: PIXELS=\"$pixels\" bash scripts/eval_pixel_sweep.sh $kind $ckpt $task ${TAG}_${label}"
        continue
    fi
    # `set -e` + `pipefail` would abort the whole sweep if one arm dies; each
    # point set is independent, so isolate the failure and keep going.
    if ! { PIXELS="$pixels" bash scripts/eval_pixel_sweep.sh \
            "$kind" "$ckpt" "$task" "${TAG}_${label}" 2>&1 | tee "$LOG"; }; then
        echo "[0906-res] FAILED point set: $label / $task — continuing" >&2
    fi
done

echo
echo "[0906-res] DONE tier=$TIER"
echo "[0906-res] next: latency for the matching configs —"
echo "  python test_inference_bench.py --model-family qwen3_5 --tasks pareto \\"
echo "    --pareto-tokens 256,640,1280,1764 --hr-cap $HR_CAP ..."
