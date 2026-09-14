#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 0911 xattn off_ksize chain (FOREGROUND, platform entry command)
# =============================================================================
# Gate question: does question_inject='xattn' make the offset sampling points
# move with question CONTENT (lift the prior semantics-blind verdict)?
#
# Three fresh base -> CPT -> SFT lineages, off_ksize in {3,5,7}, SERIAL on one
# 8-GPU pod, k=3 first (baseline + primary "does xattn work" readout):
#   1. CPT  k=3   ->  0911_pt_qwen35_4b_xattn_k3    then  SFT -> 0911_sft_..._k3
#   2. CPT  k=5   ->  0911_pt_qwen35_4b_xattn_k5    then  SFT -> 0911_sft_..._k5
#   3. CPT  k=7   ->  0911_pt_qwen35_4b_xattn_k7    then  SFT -> 0911_sft_..._k7
#
# Each lineage sets use_intention_branch=False + use_spatial_attn_guide=False so
# the QuestionReadout cross-attn residual (modeling ~L1226) is the SOLE
# question->offset spatial path (the pre-xattn spatial_guide multiply at L1214 is
# gated by `use_intention_branch AND use_spatial_attn_guide`). off_grps=8,
# grid_size=20, inter_size=128 held; only off_ksize sweeps.
#
# Run as the platform task command (no nohup — the platform kills the pod when
# the entry process exits):
#   bash /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/qwen3_5_xattn/run_chain_xattn_ksize.sh
#
# Resume after a crash: pass KSIZES to drop already-finished kernels, e.g.
#   KSIZES="5 7" bash .../run_chain_xattn_ksize.sh
#
# Output goes to stdout (platform log collector) AND per-stage files on CFS.
# =============================================================================

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

KSIZES="${KSIZES:-3 5 7}"
XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
LOG_DIR="${LOG_DIR:-$XZF_ROOT/vldat_experiments/logs_0911}"
export XZF_ROOT CKPT_ROOT
STAMP="$(date +%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

GEN="scripts/qwen3_5_xattn"
PT_SRC="scripts/qwen3_5_0901/exp_pretrain_qwen35_4b_dat_nogate.sh"
SFT_SRC="scripts/qwen3_5_0901/exp_sft_qwen35_4b_dat_ivcap.sh"
MODELING="llava/model/language_model/modeling_qwen3_5_dat.py"
HARNESS="llava/train/train_qwen_dat.py"

# GPU/CPU telemetry sidecar: 60s snapshots to CFS, so utilization can be
# inspected from the dev machine (pods aren't directly reachable over ssh).
HOST_TAG="$(hostname -s 2>/dev/null || echo pod)"
GPU_STATS="$LOG_DIR/${HOST_TAG}_${STAMP}_xattn_gpu.csv"
CPU_STATS="$LOG_DIR/${HOST_TAG}_${STAMP}_xattn_cpu.log"
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
GPU_STATS_PID=$!
trap 'kill $GPU_STATS_PID 2>/dev/null' EXIT

banner() { echo; echo "########## [$(date '+%F %T')] $1 ##########"; echo; }

# ============================================================================
# SETUP (once): guard modeling, patch harness, derive per-k CPT/SFT scripts
# ============================================================================
banner "SETUP 0/3: guard modeling has xattn code"
if ! grep -q "self.q_readout" "$MODELING" 2>/dev/null; then
    echo "[FATAL] $MODELING has no 'self.q_readout' — rsync the local (xattn) modeling to C first." >&2
    echo "        Refusing to launch: --dat_question_inject xattn would be silently ignored -> broken run." >&2
    exit 1
fi
if ! grep -q "question_inject" "$MODELING" 2>/dev/null; then
    echo "[FATAL] $MODELING has no 'question_inject' handling — rsync the xattn modeling to C first." >&2
    exit 1
fi
if grep -Fq "sample_locs[..., (1, 0)]" "$MODELING"; then
    echo "[FATAL] $MODELING still swaps (x,y) before grid_sample — apply the coordinate fix." >&2
    exit 1
fi
echo "[ok] modeling xattn + coordinate guards passed"

banner "SETUP 1/3: patch training harness (idempotent, fail-loud)"
python3 - "$HARNESS" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text(); orig = s; changed = []

# 1a) DAT_KEYS_MATCH must include q_readout (else CPT freeze-unfreeze AND SFT LoRA
#     both leave QuestionReadout frozen at random init).
if "q_readout" not in s:
    anchor = "'conv_lr_dw', 'ln_1', 'conv_lr_proj', 'proj_intention',"
    if anchor not in s:
        sys.exit("PATCH FAIL: DAT_KEYS_MATCH anchor not found — inspect harness on C")
    s = s.replace(anchor, anchor + " 'q_readout',", 1); changed.append("DAT_KEYS_MATCH+=q_readout")

# 1b) 3 dataclass fields after dat_off_ksize
if "dat_question_inject" not in s:
    anchor = "    dat_off_ksize: int = field(default=3)\n"
    if anchor not in s:
        sys.exit("PATCH FAIL: dat_off_ksize dataclass anchor not found")
    add = ('    dat_question_inject: str = field(default="none", metadata={"help": '
           '"none|xattn: question-conditioned offset readout (QuestionReadout)"})\n'
           "    dat_qr_heads: int = field(default=4)\n"
           "    dat_qr_layerscale_init: float = field(default=1e-2)\n")
    s = s.replace(anchor, anchor + add, 1); changed.append("dataclass+=3 qr fields")

# 1c) 3 dat_extra_args entries after 'off_ksize'
if "'question_inject':" not in s:
    anchor = "            'off_ksize': model_args.dat_off_ksize,\n"
    if anchor not in s:
        sys.exit("PATCH FAIL: dat_extra_args off_ksize anchor not found")
    add = ("            'question_inject': model_args.dat_question_inject,\n"
           "            'qr_heads': model_args.dat_qr_heads,\n"
           "            'qr_layerscale_init': model_args.dat_qr_layerscale_init,\n")
    s = s.replace(anchor, anchor + add, 1); changed.append("dat_extra_args+=3 qr entries")

if s != orig:
    p.write_text(s); print("HARNESS PATCHED:", ", ".join(changed))
else:
    print("HARNESS already patched (no change)")
PY

banner "SETUP 2/3: derive per-k CPT/SFT scripts from the 0901 originals"
# Only touches the dat-arg lines; reuses proven env/preflight/staging/merge.
# Copies live at scripts/qwen3_5_xattn/ (same 2-levels depth as 0901) so every
# BASH_SOURCE-relative path inside them still resolves to the repo root.
patch_script() {  # $1=src $2=dst $3=ksize
    python3 - "$1" "$2" "$3" <<'PY'
import sys
src, dst, k = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src).read()
assert "--dat_use_intention_branch True" in s, f"intention_branch flag missing in {src}"
assert "--dat_intention_as_gate True" in s, f"intention_as_gate flag missing in {src}"
s = s.replace("--dat_use_intention_branch True", "--dat_use_intention_branch False")
s = s.replace("--dat_intention_as_gate True",    "--dat_intention_as_gate False")
anchor = "    --dat_hd_proj True \\\n"
assert anchor in s, f"--dat_hd_proj anchor missing in {src}"
add = (f"    --dat_off_ksize {k} \\\n"
       "    --dat_question_inject xattn \\\n"
       "    --dat_qr_heads 4 \\\n"
       "    --dat_qr_layerscale_init 1e-2 \\\n")
s = s.replace(anchor, anchor + add, 1)
open(dst, "w").write(s)
print("generated", dst)
PY
}
for K in $KSIZES; do
    patch_script "$PT_SRC"  "$GEN/_cpt_xattn_k${K}.sh"  "$K"
    patch_script "$SFT_SRC" "$GEN/_sft_xattn_k${K}.sh"  "$K"
done
echo "[ok] derived scripts for k in: $KSIZES"

banner "SETUP 3/3: done — starting the serial chain"

# ============================================================================
# CHAIN: k=3 -> k=5 -> k=7, each CPT (foreground) then SFT (foreground)
# ============================================================================
STAGE=0
for K in $KSIZES; do
    PT_EXP="0911_pt_qwen35_4b_xattn_k${K}"
    SFT_EXP="0911_sft_qwen35_4b_xattn_k${K}"

    STAGE=$((STAGE + 1))
    banner "K=${K} CPT -> $PT_EXP"
    EXP_NAME="$PT_EXP" bash "$GEN/_cpt_xattn_k${K}.sh" 2>&1 \
        | tee "$LOG_DIR/xattn_${STAMP}_k${K}_1_cpt.log"

    CPT_OUT="$CKPT_ROOT/$PT_EXP"
    if [[ ! -f "$CPT_OUT/config.json" ]]; then
        echo "[FATAL] CPT output missing: $CPT_OUT/config.json — aborting K=$K SFT" >&2
        exit 1
    fi

    banner "K=${K} SFT -> $SFT_EXP  (from $CPT_OUT)"
    EXP_NAME="$SFT_EXP" MODEL_PATH="$CPT_OUT" bash "$GEN/_sft_xattn_k${K}.sh" 2>&1 \
        | tee "$LOG_DIR/xattn_${STAMP}_k${K}_2_sft.log"

    banner "K=${K} DONE (CPT+SFT)"
    echo "stage-1 ckpt: $PT_EXP"
    echo "stage-2 ckpt: $SFT_EXP"
done

banner "xattn off_ksize chain DONE: k in $KSIZES"
