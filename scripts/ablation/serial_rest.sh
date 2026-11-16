#!/usr/bin/env bash
# One chain, one machine: smokes first, then the long runs.
#   1. Ablation 3 native SFT, four HR-Bench 4K budgets (the Q-LAS rows are done).
#   2. Ablation 4, loss rows.
#   3. The 4B density head (train, then V* / HR-Bench 4K / 8K).
# Ablation 5 is not in this chain.
# A failed smoke stops the chain before any long run. Finished points and heads
# are skipped by the underlying scripts, so a relaunch resumes.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export SKIP_LOCK=1
export PORT=30820

W4=$M4B/where_lm8_ce20.pt
need "$M4B" "$W4" "$QA" "$HOME/sft_data/train_split" "$HOME/lmms-eval"
grep -q '_limit_image' "$HOME/lmms-eval/lmms_eval/models/simple/qwen3_vl.py" \
    || { say "STOP ~/lmms-eval has no _limit_image; pull that repo first"; exit 1; }

die() {
    say "STOP $*"
    exit 1
}

# die_log REASON LOG
die_log() {
    say "STOP $1 (see $2)"
    tail -n 30 "$2" | while IFS= read -r line; do say "  $line"; done
    exit 1
}

# check_caps LOG EXPECTED_MAX_PIXELS
# EXPECTED is a space-separated list. Every logged cap must be one of them,
# under that pixel budget, and away from the 4032 full-res image. A launch
# banner ([base35_tokN]) must have a matching cap line; skipped points
# (already done) have neither.
check_caps() {
    python - "$1" "$2" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
want = [int(x) for x in sys.argv[2].split()]
starts = [int(x) for x in re.findall(r"\[base35_tok(\d+)\]", text)]
raw = re.findall(r"image pixel cap (\d+)\.\.(\d+) -> (\d+)x(\d+)", text)
parsed = []
for mn, cap, w, h in raw:
    cap, w, h = int(cap), int(w), int(h)
    if cap not in want or w * h > cap or max(w, h) >= 3000:
        sys.exit(f"bad cap {mn}..{cap} -> {w}x{h}")
    item = (cap, w, h)
    if not parsed or parsed[-1] != item:
        parsed.append(item)
if len(parsed) < len(starts):
    sys.exit(f"cap lines {len(parsed)} < launches {len(starts)}")
if len({(w, h) for _, w, h in parsed}) != len(parsed):
    sys.exit(f"resized sizes not distinct: {parsed}")
print(" ".join(f"{c}:{w}x{h}" for c, w, h in parsed) or "no new launches")
PY
}

# has_results DIR
has_results() {
    find "$1" -name '*results.json' -print -quit 2>/dev/null | grep -q .
}

# smoke_eval NAME [env assignments...]
smoke_eval() {
    local name=$1
    shift
    local out=/tmp/abl_smoke_$name log=$L/smoke_$name.log
    rm -rf "$out"
    say "smoke eval $name"
    env MODEL_PATH="$M2" WHERE_PT="$W2" LDAT_PT="$H2" CROP_TOK=400 N_WIN=1 GRID=0 \
        TOKS=640 DAT_ALPHA=0.8 DAT_K=0.6 DAT_RATIO=0 WIN_FULL=0 WHERE_LAYER=8 \
        FAST_PREFIX=1 PREFIX_CACHE=0 SAMPLE_FEAT=0 TASKS=hrbench4k LIMIT=8 \
        PORT="$PORT" OUT="$out" "$@" \
        bash scripts/eval_ldat_pareto.sh > "$log" 2>&1 || die_log "eval $name" "$log"
    PORT=$((PORT + 10))
    grep -q '\[FAIL\]' "$log" && die_log "eval $name" "$log"
    has_results "$out" || die_log "eval $name produced no results" "$log"
    say "smoke eval $name ok"
}

# smoke_train NAME MODEL WHERE "loss flags"
# One step, eight GPUs, same recipe as the real train. The checkpoint stays in /tmp.
smoke_train() {
    local name=$1 model=$2 where=$3 loss=$4
    local out=/tmp/abl_smoke_${name}.pt log=$L/smoke_$name.log
    rm -f "$out"
    say "smoke train $name"
    # shellcheck disable=SC2086
    torchrun --nproc_per_node 8 --master_port "$PORT" scripts/offset_feat_train.py \
        --model_path "$model" --where_head_pt "$where" --n_win 1 --crop_tok 400 \
        --pe global --ldat_in native --ldat_d 256 --ldat_win dat --src_scale 4 \
        --dat_alpha 0.8 --dat_ratio 0 --image_root "$HOME/sft_data/train_split" \
        --held_from_train 8 --held_eval_n 1 --max_train 32 --accum 4 \
        --eval_every 1000 --steps 1 --train_json "$QA" --cfgs crop,dat,ldat \
        --train_cfg ldat --lr 1e-4 $loss --save "$out" > "$log" 2>&1 \
        || die_log "train $name" "$log"
    PORT=$((PORT + 10))
    [[ -s "$out" ]] || die_log "train $name saved nothing" "$log"
    grep -q "EVAL step 1" "$log" || die_log "train $name did not finish a step" "$log"
    say "smoke train $name ok"
}

report() {
    python - "$A" <<'PY' | while IFS= read -r line; do say "$line"; done
import glob, json, os, sys
root = sys.argv[1]

def emit(row, f):
    try:
        res = json.load(open(f)).get("results", {})
    except Exception:
        return
    tok = next((p for p in reversed(f.split(os.sep)) if "tok" in p or "_lr" in p),
               os.path.basename(os.path.dirname(f)))
    for key, metrics in res.items():
        if not isinstance(metrics, dict):
            continue
        for k, v in sorted(metrics.items()):
            if k.endswith(",none") and isinstance(v, (int, float)):
                shown = v * 100 if v <= 1 else v
                print(f"SCORE {row} {key} {tok} {k.replace(',none','')} {shown:.2f}")

for d in sorted(glob.glob(root + "/native_sft_cap/*")):
    for f in glob.glob(d + "/**/*_results.json", recursive=True):
        emit("native", f)
        break
for row in sorted(os.listdir(root)):
    if not (row.startswith("loss_") or row == "d4b_qlas"):
        continue
    base = os.path.join(root, row)
    if not os.path.isdir(base):
        continue
    for f in glob.glob(base + "/**/*_results.json", recursive=True):
        emit(row, f)
PY
}

head4=$A/heads/ldat4b_hinge_noratio.pt
if [[ -s "$head4" ]]; then
    aside="${head4}.aside.$(date +%s)"
    say "4B head already on disk, moving it aside so this chain trains a new one: $aside"
    mv "$head4" "$aside" || die "could not move $head4 aside"
fi

for f in hinge_ldat ctrl2_ldat both_ldat; do
    if [[ -s "$X/$f.pt" ]]; then say "abl4 will score $f"; else say "abl4 will skip $f (file missing)"; fi
done

say "smokes, then native SFT, ablation 4, 4B head"

# --- smokes. Any failure exits before the long runs. ---
out=/tmp/abl_smoke_native
log=$L/smoke_native.log
rm -rf "$out"
say "smoke native SFT"
env LMMS_EVAL_DIR=$HOME/lmms-eval PIXELS=655360 OUT_ROOT=$out BASE_AS_QWEN=1 \
    PORT=$PORT LIMIT=8 CONDA_ENV="${CONDA_ENV:-fastvlm}" \
    bash scripts/eval_pixel_sweep.sh base35 "$M2" hrbench4k smoke_native \
    > "$log" 2>&1 || die_log "native smoke" "$log"
PORT=$((PORT + 10))
[[ -f $out/base35_tok640/done ]] || die_log "native smoke" "$log"
if ! msg=$(check_caps "$log" "655360" 2>&1); then
    say "STOP native cap: $msg"
    die_log "native cap" "$log"
fi
say "smoke native ok $msg"

smoke_eval abl4
r4=""
for f in hinge_ldat ctrl2_ldat both_ldat; do
    [[ -s "$X/$f.pt" ]] && r4=$X/$f.pt && break
done
if [[ -n "$r4" ]]; then
    smoke_eval abl4_ratio4 LDAT_PT="$r4" DAT_RATIO=4
else
    say "smoke eval abl4_ratio4 skipped (no ratio-4 head on disk)"
fi

smoke_train loss2b "$M2" "$W2" "$FULL_LOSS"
smoke_train loss4b "$M4B" "$W4" "$FULL_LOSS"
smoke_eval d4b MODEL_PATH="$M4B" WHERE_PT="$W4" LDAT_PT=/tmp/abl_smoke_loss4b.pt
rm -f /tmp/abl_smoke_loss2b.pt /tmp/abl_smoke_loss4b.pt
say "smokes passed"

# --- long runs ---
log=$L/native_sft_cap.log
say "full native SFT"
env LMMS_EVAL_DIR=$HOME/lmms-eval \
    PIXELS="655360 1064960 1720320 3031040" OUT_ROOT=$A/native_sft_cap \
    BASE_AS_QWEN=1 PORT=$PORT CONDA_ENV="${CONDA_ENV:-fastvlm}" \
    bash scripts/eval_pixel_sweep.sh base35 "$M2" hrbench4k native_sft_cap \
    > "$log" 2>&1 || die_log "native SFT" "$log"
PORT=$((PORT + 10))
for tok in 640 1040 1680 2960; do
    [[ -f $A/native_sft_cap/base35_tok${tok}/done ]] || die_log "native tok $tok" "$log"
done
if ! msg=$(check_caps "$log" "655360 1064960 1720320 3031040" 2>&1); then
    say "STOP native cap: $msg"
    die_log "native cap" "$log"
fi
say "native SFT ok $msg"

say "full ablation 4"
PORT=$((PORT + 20))
bash scripts/ablation/abl4_loss.sh > "$L/abl4_loss.out" 2>&1 || die_log "ablation 4" "$L/abl4_loss.out"
for n in loss_no_hinge loss_no_box loss_no_tr; do
    [[ -s $A/heads/$n.pt ]] || die "ablation 4 missing $A/heads/$n.pt"
    has_results "$A/$n" || die "ablation 4 missing scores for $n"
done
for pair in r4_hinge:hinge_ldat r4_ctr:ctrl2_ldat r4_hinge_ctr:both_ldat; do
    f=$X/${pair#*:}.pt
    [[ -s "$f" ]] || continue
    has_results "$A/loss_${pair%%:*}" || die "ablation 4 missing scores for loss_${pair%%:*}"
done
say "ablation 4 ok"

say "full 4B head"
PORT=$((PORT + 40))
bash scripts/ablation/main_4b_density.sh > "$L/main_4b.out" 2>&1 || die_log "4B head" "$L/main_4b.out"
[[ -s $head4 ]] || die "4B head was not saved"
for tok in 640 1280; do
    for task in vstar_bench hrbench4k hrbench8k; do
        has_results "$A/d4b_qlas/${task}_lr${tok}" || die "4B missing $task lr $tok"
    done
done
say "4B head ok"
report
say "serial done"
