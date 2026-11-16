# Shared settings of the Q-LAS ablations, sourced by every script in this folder.
#
# Every row is the 2B main setting unless its script overrides it:
#   checkpoint-1000, where-head on LM layer 8 (where_lm8_ce20.pt), trained density
#   head (hinge_noratio_ldat.pt), LR 640 + one 400-token window,
#   scored on HR-Bench 4K (800 questions) by lmms-eval.
# Rows: $A/<row>/<task>_lr<N>/   heads: $A/heads/   logs: $L/<row>.log   timeline: $L/chain.log
# A finished row or head is skipped, so a script can be relaunched after a failure.
# Scripts on one machine take turns through $L/lock: launch several at once and
# they run one after another.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CONDA_SH=${CONDA_SH:-$HOME/miniconda3/bin/activate}
export CONDA_ENV=${CONDA_ENV:-fastvlm}
# shellcheck disable=SC1090
source "$CONDA_SH" "$CONDA_ENV"

E=${E:-$HOME/vldat_experiments/0924_miniE1_2k}
M4B=${M4B:-$HOME/vldat_experiments/1110_4b_viscot_lora-merged}
X=${X:-/data/oss_bucket_0/wangziyi/vldat_experiments/sa1b_ldat}
OSS_DATA=${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}
A=${A:-$X/ablation}
L=${L:-/tmp/ablation}
PORT=${PORT:-30410}
TASKS_ABL=${TASKS_ABL:-hrbench4k}
export LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-$HOME/lmms-eval-oldvstar}
export HRBENCH_SKIP_GPT_EVAL=1 PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# eval_ldat_pareto.sh reads these; a value left in the shell would cut every row short.
unset LIMIT NPROC

M2=$E/checkpoint-1000
W2=$E/loc/where_lm8_ce20.pt
H2=$X/hinge_noratio_ldat.pt
Z2=$A/heads/ldat_zero_2b.pt
QA=$X/data/sa1b_small_qa.json
mkdir -p "$A/heads" "$L"

NAME=$(basename "$0" .sh)
say() { echo "$(date '+%F %T') [$NAME] $*" | tee -a "$L/chain.log"; }

need() {
    local p
    for p in "$@"; do
        [ -e "$p" ] || { say "missing $p"; exit 1; }
    done
}
need "$M2" "$W2" "$H2" "$LMMS_EVAL_DIR"

exec 9>"$L/lock"
say "waiting for the GPUs"
flock 9
say "start"

# ev ROW [VAR=value ...]: one eval_ldat_pareto.sh run of the main setting with the overrides.
ev() {
    local row=$1
    shift
    say "eval $row $*"
    env MODEL_PATH="$M2" WHERE_PT="$W2" LDAT_PT="$H2" CROP_TOK=400 N_WIN=1 GRID=0 TOKS=640 \
        DAT_ALPHA=0.8 DAT_K=0.6 DAT_RATIO=0 WIN_FULL=0 WHERE_LAYER=8 \
        FAST_PREFIX=1 PREFIX_CACHE=0 SAMPLE_FEAT=0 TASKS="$TASKS_ABL" PORT="$PORT" \
        "$@" OUT="$A/$row" bash scripts/eval_ldat_pareto.sh >> "$L/$row.log" 2>&1
    PORT=$((PORT + 10))
}

# mkzero OUT MODEL: density head with a zero residual, so the density is the where-head's.
mkzero() {
    [ -f "$1" ] && return 0
    # config.json is read directly: the SFT checkpoint's model_type (qwen3_5_dat)
    # is unknown to AutoConfig.
    python - "$1" "$2" <<'PY'
import json
import os
import sys
sys.path.insert(0, "scripts")
import torch
from offset_feat_train import LDatHead
cfg = json.load(open(os.path.join(sys.argv[2], "config.json")))
hidden = cfg.get("text_config", cfg)["hidden_size"]
tmp = f"{sys.argv[1]}.{os.getpid()}.tmp"
torch.save(LDatHead(hidden, d=256).state_dict(), tmp)
os.replace(tmp, sys.argv[1])
print("[zero]", sys.argv[1], "hidden", hidden)
PY
    [ -f "$1" ] || { say "zero density head FAILED for $2"; exit 1; }
}

# The recipe of hinge_noratio_ldat.pt without its loss weights.
train_base() {   # train_base MODEL WHERE
    echo "--model_path $1 --where_head_pt $2 --n_win 1 --crop_tok 400 --pe global --ldat_in native" \
         "--ldat_d 256 --ldat_win dat --src_scale 4 --dat_alpha 0.8 --dat_ratio 0" \
         "--image_root $HOME/sft_data/train_split --held_from_train 2000 --held_eval_n 400" \
         "--max_train 60000 --accum 4 --eval_every 250 --save_best --steps 2000" \
         "--train_json $QA --cfgs crop,dat,ldat --train_cfg ldat --lr 1e-4"
}
FULL_LOSS="--ans_w 1 --box_w 0.1 --tr_w 0.1 --hinge_w 1"

train_ldat() {   # train_ldat NAME "base flags" "loss flags"
    local out=$A/heads/$1.pt
    [ -f "$out" ] && return 0
    say "train $1: $3"
    # shellcheck disable=SC2086
    torchrun --nproc_per_node 8 --master_port "$PORT" scripts/offset_feat_train.py $2 $3 \
        --save "$out" > "$L/train_$1.log" 2>&1
    PORT=$((PORT + 10))
    [ -f "$out" ] || { say "train $1 FAILED, see $L/train_$1.log"; return 1; }
}
