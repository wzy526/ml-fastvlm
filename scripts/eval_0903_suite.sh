#!/usr/bin/env bash
# 0903 eval suite: Qwen3.5 2B/4B base-vs-DAT (+ 4B early-exit arms).
#
# Six configs across two pods:
#   pod1: 2B base | 2B DAT (0902 merged)        | 4B base
#   pod2: 4B DAT (k0) | 4B DAT ee12 | 4B DAT ee18   (all 0901 merged)
#
# Two task tiers (swept by eval_pixel_sweep.sh over LLM-token budgets):
#   HR tier  — token-axis-sensitive, one launch per task, 6-point grid
#              (256/640/1280/2560/6400/11520 tokens):
#              chartqa docvqa_val textvqa_val vstar_bench hrbench4k hrbench8k
#              refcoco_bbox_rec_val_small
#   GEN tier — low-res standard suites, all 7 tasks batched into ONE lmms-eval
#              launch per pixel point (one model load), 4-point grid
#              (256/640/1280/2560 tokens; higher budgets saturate at native
#              image size):
#              gqa scienceqa_img vizwiz_vqa_val pope mmbench_en_dev seedbench mme
#
# Usage (platform entry command or interactive shell on the pod):
#   bash /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/eval_0903_suite.sh pod1
#   bash /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/eval_0903_suite.sh pod2
#
# Quick sanity pass (all configs, one task, one pixel point):
#   TASKS=chartqa PIXELS=2621440 bash .../eval_0903_suite.sh pod1
#
# Resume-safe: eval_pixel_sweep.sh skips points that already have a `done`
# marker, so rerunning the same command continues where it stopped.
#
# Env knobs:
#   TASKS       override: space list, per-task launches on the HR grid;
#               GEN batching disabled (sanity mode)
#   HR_TASKS / GEN_TASKS   override either tier's task list
#   PIXELS      one grid for everything (overrides both tier grids)
#   PIXELS_HR / PIXELS_GEN grids per tier
#   NPROC, GPUS forwarded to eval_pixel_sweep.sh
#   LOCAL_CACHE model staging dir (default /workspace/model_cache; CFS fallback)
#
# Outputs:
#   results : <repo>/_test_outputs/_sweep_<name>_<cfg>/   (on CFS; GEN name=gen7)
#   logs    : $XZF_ROOT/vldat_experiments/logs_0903_eval/ (on CFS)

set -uo pipefail   # NO -e: one failed run must not kill the whole suite.

PLAN="${1:?usage: $0 <pod1|pod2>}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO/scripts/eval_pod_env.sh"

XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
MODELS_ROOT="${MODELS_ROOT:-/home/ea-cv-nlp-train-offline-2/xzf/models}"
LOCAL_CACHE="${LOCAL_CACHE:-/workspace/model_cache}"
LOG_DIR="${LOG_DIR:-$CKPT_ROOT/logs_0903_eval}"
STAMP="$(date +%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# ---- token grids (pixels = tokens * 1024 for the qwen3.5 family) ------------
PIXELS_HR="${PIXELS_HR:-262144 655360 1310720 2621440 6553600 11796480}"
PIXELS_GEN="${PIXELS_GEN:-262144 655360 1310720 2621440}"
if [[ -n "${PIXELS:-}" ]]; then PIXELS_HR="$PIXELS"; PIXELS_GEN="$PIXELS"; fi

HR_TASKS="${HR_TASKS:-chartqa docvqa_val textvqa_val vstar_bench hrbench4k hrbench8k refcoco_bbox_rec_val_small}"
GEN_TASKS="${GEN_TASKS:-gqa scienceqa_img vizwiz_vqa_val pope mmbench_en_dev seedbench mme}"

# ---- configs per pod: "model_type|ckpt_dir|tag" ------------------------------
case "$PLAN" in
    pod1)
        CONFIGS=(
            "base35|$MODELS_ROOT/Qwen3.5-2B|0903_2b_base"
            "dat35|$CKPT_ROOT/0902_sft_qwen35_2b_dat_ivcap-merged|0903_2b_dat"
            "base35|$MODELS_ROOT/Qwen3.5-4B|0903_4b_base"
        )
        ;;
    pod2)
        CONFIGS=(
            "dat35|$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap-merged|0903_4b_dat"
            "dat35|$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap_ee12-merged|0903_4b_dat_ee12"
            "dat35|$CKPT_ROOT/0901_sft_qwen35_4b_dat_ivcap_ee18-merged|0903_4b_dat_ee18"
        )
        ;;
    *) echo "[ERROR] PLAN must be pod1|pod2, got '$PLAN'" >&2; exit 1 ;;
esac

# ---- task specs: "log_name|tasks_arg|pixel_grid" -----------------------------
# pod1 runs GEN first + HR forward; pod2 runs HR reversed + GEN last. The two
# pods therefore never build the same HF arrow cache (first dataset load) at
# the same time on shared CFS.
TASK_SPECS=()
if [[ -n "${TASKS:-}" ]]; then
    for t in $TASKS; do TASK_SPECS+=("$t|$t|$PIXELS_HR"); done
else
    GEN_CSV="${GEN_TASKS// /,}"
    HR_FWD=(); for t in $HR_TASKS; do HR_FWD+=("$t|$t|$PIXELS_HR"); done
    if [[ "$PLAN" == "pod1" ]]; then
        TASK_SPECS=("gen7|$GEN_CSV|$PIXELS_GEN" "${HR_FWD[@]}")
    else
        for ((i = ${#HR_FWD[@]} - 1; i >= 0; i--)); do TASK_SPECS+=("${HR_FWD[$i]}"); done
        TASK_SPECS+=("gen7|$GEN_CSV|$PIXELS_GEN")
    fi
fi

# ---- fail fast on missing inputs --------------------------------------------
FAIL=0
for spec in "${CONFIGS[@]}"; do
    IFS='|' read -r _ ckpt _ <<< "$spec"
    [[ -f "$ckpt/config.json" ]] || { echo "[ERROR] missing ckpt: $ckpt" >&2; FAIL=1; }
done
[[ -d "$HF_HOME/hub" ]] || { echo "[ERROR] HF_HOME/hub missing: $HF_HOME/hub" >&2; FAIL=1; }
[[ -d "$LMMS_EVAL_DIR" ]] || { echo "[ERROR] LMMS_EVAL_DIR missing: $LMMS_EVAL_DIR" >&2; FAIL=1; }
if (( FAIL )); then exit 1; fi

# ---- GPU/CPU telemetry sidecar (same pattern as the training chains) --------
HOST_TAG="$(hostname -s 2>/dev/null || echo pod)"
GPU_STATS="$LOG_DIR/${PLAN}_${HOST_TAG}_${STAMP}_gpu.csv"
CPU_STATS="$LOG_DIR/${PLAN}_${HOST_TAG}_${STAMP}_cpu.log"
echo "ts,idx,util_pct,mem_used_mib,mem_total_mib,power_w" >> "$GPU_STATS"
(
    while true; do
        TS="$(date '+%F %T')"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk -v t="$TS" -F', *' '{OFS=","; print t,$1,$2,$3,$4,$5}' >> "$GPU_STATS"
        echo "$TS loadavg=$(cut -d' ' -f1-3 /proc/loadavg) nproc=$(nproc)" >> "$CPU_STATS"
        sleep 60
    done
) &
SIDE_PID=$!
trap 'kill $SIDE_PID 2>/dev/null' EXIT

banner() { echo; echo "########## [$(date '+%F %T')] $1 ##########"; echo; }

# ---- stage models from CFS to pod-local disk (each config is loaded once per
#      lmms-eval launch x 8 ranks x every pixel point — local disk pays off) --
stage() {  # $1 = src dir -> echoes local path (or src on any failure)
    local src="$1" name dst
    name="$(basename "$src")"
    dst="$LOCAL_CACHE/$name"
    if [[ -f "$dst/config.json" ]]; then echo "$dst"; return; fi
    if mkdir -p "$LOCAL_CACHE" 2>/dev/null \
        && rsync -a --exclude='checkpoint-*' --exclude='tb' "$src/" "$dst.tmp/" 2>/dev/null \
        && rm -rf "$dst" && mv "$dst.tmp" "$dst"; then
        echo "$dst"
    else
        echo "[stage WARN] could not stage $name to $LOCAL_CACHE; using CFS path" >&2
        rm -rf "$dst.tmp" 2>/dev/null
        echo "$src"
    fi
}

banner "staging models to $LOCAL_CACHE"
STAGED=()
for spec in "${CONFIGS[@]}"; do
    IFS='|' read -r mtype ckpt tag <<< "$spec"
    t0=$SECONDS
    local_ckpt="$(stage "$ckpt")"
    echo "[stage] $tag: $local_ckpt ($((SECONDS - t0))s)"
    STAGED+=("$mtype|$local_ckpt|$tag")
done

# ---- main loop: task-major so a dataset's arrow cache is built only once ----
banner "plan=$PLAN"
for spec in "${TASK_SPECS[@]}"; do echo "  spec:   ${spec%%|*} (grid: ${spec##*|})"; done
for spec in "${STAGED[@]}"; do echo "  config: $spec"; done

RUN_IDX=0
SUMMARY=()
for tspec in "${TASK_SPECS[@]}"; do
    IFS='|' read -r name tasks_arg grid <<< "$tspec"
    for cspec in "${STAGED[@]}"; do
        IFS='|' read -r mtype ckpt tag <<< "$cspec"
        banner "task=$name  config=$tag  ($mtype)"
        log="$LOG_DIR/${PLAN}_${STAMP}_${name}_${tag}.log"
        # Distinct port base per run: a crashed launcher can leave the previous
        # port in TIME_WAIT. OUT_ROOT pinned here so comma lists get a sane dir.
        if PIXELS="$grid" \
            PORT=$((31000 + RUN_IDX * 100)) \
            OUT_ROOT="$REPO/_test_outputs/_sweep_${name}_${tag}" \
            bash "$REPO/scripts/eval_pixel_sweep.sh" "$mtype" "$ckpt" "$tasks_arg" "$tag" \
            2>&1 | tee "$log"; then
            SUMMARY+=("OK    $name $tag")
        else
            SUMMARY+=("FAIL  $name $tag (see $(basename "$log"))")
        fi
        RUN_IDX=$((RUN_IDX + 1))
    done
done

banner "$PLAN suite DONE"
printf '%s\n' "${SUMMARY[@]}"
echo
echo "results: $REPO/_test_outputs/_sweep_<name>_<cfg>/"
echo "logs:    $LOG_DIR/${PLAN}_${STAMP}_*.log"
