#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# 0913 xattn EE6 chain: k=3/5/7 with HD ViT early-exit at block 6,
# off_range=0, off_penalty=0 (plain clamp in the default branch).
#
# Differences vs run_chain_xattn_ksize.sh:
#   - --dat_hd_early_exit_k 6  (HD ViT truncated to first 6/24 blocks;
#     0901 ee-sweep showed ee6 is accuracy-acceptable)
#   - off_range / off_penalty stay at their 0901 defaults (0/0) — the
#     modeling file's default branch is now a plain clamp (no straight-
#     through), so out-of-bounds points get zero gradient instead of a
#     further-out push.
#   - fresh base -> CPT -> SFT per k, serial, same as before.
#
# Run as the platform task command:
#   bash /home/ea-cvfa-aigc-x2v-2/xzf/ml-fastvlm/scripts/qwen3_5_xattn/run_chain_xattn_ee6.sh
#
# Resume: KSIZES="5 7" bash .../run_chain_xattn_ee6.sh
# =============================================================================

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

KSIZES="${KSIZES:-3 5 7}"
XZF_ROOT="${XZF_ROOT:-/home/ea-cvfa-aigc-x2v-2/xzf}"
CKPT_ROOT="${CKPT_ROOT:-$XZF_ROOT/vldat_experiments}"
LOG_DIR="${LOG_DIR:-$XZF_ROOT/vldat_experiments/logs_0913_ee6}"
export XZF_ROOT CKPT_ROOT
STAMP="$(date +%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

GEN="scripts/qwen3_5_xattn"
PT_SRC="scripts/qwen3_5_0901/exp_pretrain_qwen35_4b_dat_nogate.sh"
SFT_SRC="scripts/qwen3_5_0901/exp_sft_qwen35_4b_dat_ivcap.sh"
MODELING="llava/model/language_model/modeling_qwen3_5_dat.py"
HARNESS="llava/train/train_qwen_dat.py"

# GPU/CPU telemetry sidecar (same as the ksize chain).
HOST_TAG="$(hostname -s 2>/dev/null || echo pod)"
GPU_STATS="$LOG_DIR/${HOST_TAG}_${STAMP}_ee6_gpu.csv"
CPU_STATS="$LOG_DIR/${HOST_TAG}_${STAMP}_ee6_cpu.log"
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
banner "SETUP 0/3: guard modeling has xattn code + plain clamp"
if ! grep -q "self.q_readout" "$MODELING" 2>/dev/null; then
    echo "[FATAL] $MODELING has no 'self.q_readout' — rsync the local (xattn) modeling to C first." >&2
    exit 1
fi
if ! grep -q "question_inject" "$MODELING" 2>/dev/null; then
    echo "[FATAL] $MODELING has no 'question_inject' — wrong modeling version." >&2
    exit 1
fi
# Guard: the default (off_range=0, off_penalty=0) branch must be a PLAIN
# clamp, not the old straight-through one.
if grep -q "sample_locs = (x + (x.clamp(-1, 1) - x).detach())" "$MODELING"; then
    echo "[FATAL] $MODELING still has the straight-through clamp in the default branch." >&2
    echo "        Apply the plain-clamp edit before launching this chain." >&2
    exit 1
fi
if ! grep -q "sample_locs = x.clamp(-1, 1).permute(0, 2, 3, 1)" "$MODELING"; then
    echo "[FATAL] plain clamp not found in $MODELING — verify the default branch." >&2
    exit 1
fi
if grep -Fq "sample_locs[..., (1, 0)]" "$MODELING"; then
    echo "[FATAL] $MODELING still swaps (x,y) before grid_sample — apply the coordinate fix." >&2
    exit 1
fi

banner "SETUP 1/3: patch harness (question_inject / qr_heads / qr_layerscale_init)"
python3 - "$HARNESS" <<'PY'
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "llava/train/train_qwen_dat.py"
s = orig = open(p).read()
changed = []

if "dat_question_inject" not in s:
    anchor = "    dat_off_ksize: int = field(default=3)\n"
    if anchor not in s:
        sys.exit("PATCH FAIL: dat_off_ksize dataclass anchor not found")
    add = ('    dat_question_inject: str = field(default="none", metadata={"help": '
           '"none|xattn: question-conditioned offset readout (QuestionReadout)"})\n'
           "    dat_qr_heads: int = field(default=4)\n"
           "    dat_qr_layerscale_init: float = field(default=1e-2)\n")
    s = s.replace(anchor, anchor + add, 1); changed.append("dataclass+=3 qr fields")

if "'question_inject':" not in s:
    anchor = "            'off_ksize': model_args.dat_off_ksize,\n"
    if anchor not in s:
        sys.exit("PATCH FAIL: dat_extra_args off_ksize anchor not found")
    add = ("            'question_inject': model_args.dat_question_inject,\n"
           "            'qr_heads': model_args.dat_qr_heads,\n"
           "            'qr_layerscale_init': model_args.dat_qr_layerscale_init,\n")
    s = s.replace(anchor, anchor + add, 1); changed.append("dat_extra_args+=3 qr entries")

if s != orig:
    p_write = sys.argv[1] if len(sys.argv) > 1 else "llava/train/train_qwen_dat.py"
    open(p_write, "w").write(s); print("HARNESS PATCHED:", ", ".join(changed))
else:
    print("HARNESS already patched (no change)")
PY

banner "SETUP 2/3: derive per-k EE6 CPT/SFT scripts from the 0901 originals"
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
# EE6: HD ViT early exit at block 6 (plain replace; both scripts default to 0).
assert "--dat_hd_early_exit_k 0" in s, f"hd_early_exit_k anchor missing in {src}"
s = s.replace("--dat_hd_early_exit_k 0", "--dat_hd_early_exit_k 6")
open(dst, "w").write(s)
print("generated", dst)
PY
}
for K in $KSIZES; do
    patch_script "$PT_SRC"  "$GEN/_cpt_xattn_ee6_k${K}.sh"  "$K"
    patch_script "$SFT_SRC" "$GEN/_sft_xattn_ee6_k${K}.sh"  "$K"
done
echo "[ok] derived EE6 scripts for k in: $KSIZES"

banner "SETUP 3/3: done — starting the serial chain"

# ============================================================================
# CHAIN: k=3 -> k=5 -> k=7, each CPT (foreground) then SFT (foreground)
# ============================================================================
STAGE=0
for K in $KSIZES; do
    PT_EXP="0913_pt_qwen35_4b_xattn_ee6_k${K}"
    SFT_EXP="0913_sft_qwen35_4b_xattn_ee6_k${K}"

    STAGE=$((STAGE + 1))
    banner "K=${K} CPT (EE6) -> $PT_EXP"
    EXP_NAME="$PT_EXP" bash "$GEN/_cpt_xattn_ee6_k${K}.sh" 2>&1 \
        | tee "$LOG_DIR/xattn_ee6_${STAMP}_k${K}_1_cpt.log"

    CPT_OUT="$CKPT_ROOT/$PT_EXP"
    if [[ ! -f "$CPT_OUT/config.json" ]]; then
        echo "[FATAL] CPT output missing: $CPT_OUT/config.json — aborting K=$K SFT" >&2
        exit 1
    fi

    banner "K=${K} SFT (EE6) -> $SFT_EXP  (from $CPT_OUT)"
    EXP_NAME="$SFT_EXP" MODEL_PATH="$CPT_OUT" bash "$GEN/_sft_xattn_ee6_k${K}.sh" 2>&1 \
        | tee "$LOG_DIR/xattn_ee6_${STAMP}_k${K}_2_sft.log"

    banner "K=${K} DONE (CPT+SFT, EE6)"
    echo "stage-1 ckpt: $PT_EXP"
    echo "stage-2 ckpt: $SFT_EXP"
done

banner "xattn EE6 chain DONE: k in $KSIZES"
