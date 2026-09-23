#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-fastvlm}"

# 0915 Stage-2 SFT: Qwen3.5-2B DAT (question readers + LSE-bias curriculum) + 0817 gen+VS mix.
# ============================================================================
#
# Consumes the stage-1 ckpt from exp_pretrain_qwen35_2b_dat_readers_bias.sh and
# applies the proven qwen2.5 SFT recipe on the same data as the 0817 run,
# so this run is directly comparable to 0826_sft_qwen35_2b_dat_genvs (same
# data/recipe; only the stage-1 ckpt + exact merge gradients differ):
#
#   - data: llava_hr_gen_vs_0817.json (~517k; sa1b x1.5, visualprobe x2,
#     deepeyes kept, dead OCR cut, +densefusion/allava/aokvqa/scienceqa)
#   - LLM LoRA r=8 a=16 lr=2e-5; projector (visual.merger) tuned lr=5e-6;
#     DAT lr=1e-4; ViT frozen
#   - DAT args MUST match stage 1: dat_layers auto, grid 20, nogate,
#     intention branch + as_gate ON, spatial guide OFF
#
# After training, LoRA + DAT + projector deltas are auto-merged into
# $CKPT_ROOT/$EXP_NAME-merged (merge_lora_dat_weights.py auto-detects the
# qwen3_5_dat family from the ckpt config).
#
# Teacher-forced HD sampling (bbox windows) variant — same recipe, data built by
# scripts/build_tf_bbox_data.py (VG "describe this region [box]" turns exploded
# into singles carrying `bbox`):
#   DATA_JSON=$OSS_DATA/llava_hr_gen_vs_0817_tfbox.json TF_PROB=1 \
#   EXP_NAME=0915_sft_qwen35_2b_dat_readers_bias_tfbox bash <this script>
# wandb: dat/tf_frac = fraction of samples forced per step (~0.15 expected).
#
# Offset supervision variant (0916) — the bbox windows become a regression
# TARGET for the learned sampling grid instead of replacing it, i.e. the
# intention/offset path is trained directly to find the region:
#   loss += OFF_SUP_WEIGHT * mean(huber(ref + off - window_grid))  per DAT layer
# Data: scripts/build_viscot_bbox_data.py (Visual-CoT answer-region boxes joined
# onto our docvqa/infovqa images; ~40k, HD/LR ~6x, boxes <1% of the image):
#   DATA_JSON=$OSS_DATA/llava_hr_gen_vs_0817_viscot.json TF_PROB=1 OFF_SUP_WEIGHT=10 \
#   OFF_HEAD_TRUNK_GRAD=0 EXP_NAME=0917_sft_qwen35_2b_dat_readers_bias_offsup_v2 bash <this script>
# Only the intention-conditioned slots are supervised; the question-agnostic
# image slot of QUESTION_HD=True is left alone (it cannot know the target).
# OFF_HEAD_TRUNK_GRAD=0 keeps the pull inside the offset head: at 1 (0916 run)
# it rewrote the LLM's LR image features -- offsets moved (|off| 0.08 -> 0.45,
# dist ratio 0.6 on docvqa) but HD-off V* fell 55.5 -> 50.3, DocVQA 69 -> 36.
# wandb: dat/off_sup_dist (mean point->target distance, grid units; a uniform
# grid sits ~0.5-0.8, must fall), dat/off_sup_grad_ratio (||pull||/||LM grad||
# on the grid; aim ~1-10, raise OFF_SUP_WEIGHT if << 1), dat/off_sup_loss,
# dat/offset_std (must leave the ~0.08 init floor), dat/tf_frac (= fraction of
# samples with a window, ~0.13 with the viscot mix).
#
# Global localisation variant (0917 v3) — with the pull confined to the head
# (v2), off_sup_dist STILL plateaus at ~0.5 like 0916: the per-point offset
# head is 3x3-local, a point far from the target has no information about
# which way to go. GLOBAL_OFFSET=1 adds a per-cell relevance map (1x1 conv on
# the intention-gated features, zero-init = uniform) whose soft-argmax
# translates the whole grid and whose spread shrinks it:
#   x = centroid + scale * ref + local_off
# a signal every point shares. Zero-init reproduces the legacy grid exactly,
# so it loads onto the 0915 pretrain (conv_glob is the only new param).
#   DATA_JSON=$OSS_DATA/llava_hr_gen_vs_0817_viscot.json TF_PROB=1 OFF_SUP_WEIGHT=10 \
#   OFF_HEAD_TRUNK_GRAD=0 GLOBAL_OFFSET=1 \
#   EXP_NAME=0917_sft_qwen35_2b_dat_readers_bias_offsup_glob bash <this script>
# wandb: dat/glob_shift (mean |centroid|, 0 at init, must rise), dat/glob_scale
# (mean grid scale, 1 at init; windows are ~0.3-0.4 of the image so it should
# fall toward ~0.3-0.5), and off_sup_dist should finally break below 0.4.
#
# v3 result: shift 0.17 / scale 0.76 / off_sup_dist ratio 0.78, and the probe's
# sample-dependence table shows a near-constant prior (centroid corr with the
# GT window r=0.35, |c-t| 0.47 vs 0.51 for the sample mean): the question only
# reaches the 1x1 conv as a per-channel scalar gate, which cannot do content
# matching. GLOB_REL=qk replaces the relevance source with question-cell
# matching in the trunk's hidden space (W_q(intention) . W_k(LR token), W_q
# zero-init = identity grid):
#   DATA_JSON=$OSS_DATA/llava_hr_gen_vs_0817_viscot.json TF_PROB=1 OFF_SUP_WEIGHT=10 \
#   OFF_HEAD_TRUNK_GRAD=0 GLOBAL_OFFSET=1 GLOB_REL=qk \
#   EXP_NAME=0918_sft_qwen35_2b_dat_readers_bias_offsup_qk bash <this script>
# Pass: probe sample-dependence r_cx/r_cy > 0.6, |c-t| well below const,
# scale mean approaching GT s (~0.42); then in_box should leave the 3% floor.
#
# Readout arm (0919) — the V* oracle test on v2/v3 (400 points laid on the GT
# box = perfect localisation) gave off 51.8 -> oracle 50.3 / 51.3 -> 51.3,
# while shuffle (someone else's HD) gave +2.6: the readout does not use HD
# content, so no localisation gain can reach the answer. LR dropout (per
# sample, lr_drop_ratio of the LR image tokens replaced by their mean) removes
# the LR shortcut so the LM must read the HD tokens; teacher forcing
# (OFF_SUP_WEIGHT=0 + TF_PROB=1) puts those tokens on the answer region for
# the bbox samples so what it reads is the answer:
#   DATA_JSON=$OSS_DATA/llava_hr_gen_vs_0817_viscot.json TF_PROB=1 OFF_SUP_WEIGHT=0 \
#   LR_DROP_PROB=0.5 LR_DROP_RATIO=0.75 OFF_HEAD_TRUNK_GRAD=0 \
#   EXP_NAME=0919_sft_qwen35_2b_dat_readers_bias_tforacle_lrdrop bash <this script>
# wandb: dat/lr_drop_frac (~0.5*0.75), dat/tf_frac (~0.13), kvhd_grad_norm
# (must rise vs v2). Pass: probe viscot/V* with --hd_source oracle shows
# HD-on > off by several points (first time any HD source moves accuracy).
#
# Mini readout SFT (1/20 of the above, ~1-2 h on 4-8 GPUs) — "can k/v_hd learn
# to read at all when forced": start from v2-merged, train the DAT modules ONLY
# (FREEZE_BASE=1, no LoRA, no projector), viscot bbox samples only, every
# sample teacher-forced onto its box and LR-dropped, a few hundred steps. No
# LoRA => the output dir itself is a full HF ckpt (auto-merge skips).
#   DATA_JSON=$OSS_DATA/extra_0916/viscot_bbox.train.json \
#   MODEL_PATH=~/vldat_experiments/0917_sft_qwen35_2b_dat_readers_bias_offsup_v2-merged \
#   TF_PROB=1 OFF_SUP_WEIGHT=0 LR_DROP_PROB=1 LR_DROP_RATIO=0.75 OFF_HEAD_TRUNK_GRAD=0 \
#   FREEZE_BASE=True LORA_ENABLE=False TUNE_MM_MLP=False MAX_STEPS=400 WARMUP_STEPS=20 \
#   SAVE_STEPS=200 EXP_NAME=0919_mini_readout_tforacle_lrdrop bash <this script>
# Then scripts/_test_lr_drop_leverage.py on the HELD-OUT split before/after:
# loss(S)-loss(O) must open up (oracle HD beats wrong-image HD under LR drop).
# Result: loss(O) 1.38 -> 0.94, S-O 0.06 -> 0.71, full-LR D-P 0.018 -> 0.055,
# held-out acc oracle +2.5 / shuffle -1. Readout is trainable; side effect:
# over-trust (wrong-content HD worse than none) from 100% forcing.
#
# Combined arm (0920, "B") — readout + localisation on the same data. Windows
# now do BOTH (TF_FORCE_PROB): every window supervises the learned grid
# (OFF_SUP_WEIGHT) and TF_FORCE_PROB of them also replace it. Mini version
# (DAT-only from v2, viscot train split, 600 steps) with a no-global-term
# control on the other 4 GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 DATA_JSON=$OSS_DATA/extra_0916/viscot_bbox.train.json \
#   MODEL_PATH=<v2-merged> TF_PROB=1 TF_FORCE_PROB=0.5 OFF_SUP_WEIGHT=10 LR_DROP_PROB=0.5 \
#   OFF_HEAD_TRUNK_GRAD=0 GLOBAL_OFFSET=1 GLOB_REL=qk FREEZE_BASE=True LORA_ENABLE=False \
#   TUNE_MM_MLP=False MAX_STEPS=600 WARMUP_STEPS=20 SAVE_STEPS=300 \
#   EXP_NAME=0920_miniB_qk bash <this script>
#   (control: CUDA_VISIBLE_DEVICES=4,5,6,7 ... GLOBAL_OFFSET=0 EXP_NAME=0920_miniB_noglob)
# wandb: dat/tf_frac ~1, dat/tf_forced_frac ~0.5, dat/off_sup_dist falling,
# dat/glob_shift / glob_scale moving (qk arm). Verdict: leverage test (O, S-O,
# D-P, and S <= C / A <= D for over-trust) + probe real/oracle/shuffle on the
# held-out split + the sample-dependence table (r_cx/r_cy > 0.6).
# miniB result: readout held (O 0.97, S-O 0.63, D-P 0.06), held-out real 70.5
# vs off 68.5 (first positive real), but qk localisation stayed weak (r_cx
# 0.1-0.2, in_box 3.8%). LR dropout blanks exactly what the qk relevance reads:
# half the windowed samples had no LR content to match, yet were pulled.
#
# Routing by LR dropout (ROUTE_BY_DROP=1): an LR-dropped sample is
# teacher-forced only (readout: HD must hold the answer), a full-LR sample is
# supervised (+ forced with TF_FORCE_PROB). Use it whenever LR_DROP_PROB > 0
# and OFF_SUP_WEIGHT > 0 are both on. wandb: dat/tf_sup_frac (~1 - lr_drop_prob
# of windowed samples), dat/tf_forced_frac (~lr_drop_prob + (1-lr_drop_prob) *
# TF_FORCE_PROB). If the 0920 no-drop control (LR_DROP_PROB=0) shows the readout
# does not need dropout, skip both and run LR_DROP_PROB=0 instead.
#
# nodrop control result (LR_DROP_PROB=0, else = miniB-qk): readout weaker in
# the deployment setting (D-P 0.059 -> 0.031, C-B 0.12 -> 0.007), precision
# within noise, localisation unchanged (r_cx 0.16-0.28, in_box 0.035) -- so
# dropout stays at 0.5 and the localisation problem is independent of it. Both
# runs' wandb show glob_shift jump to 0.1 in 50 steps and glob_scale to 0.9 in
# 30, then flat: the relevance map learns a constant prior and nothing else.
# The pull reaches it only through the centroid/spread of its softmax.
#
# Dense relevance supervision (REL_SUP_WEIGHT, needs GLOBAL_OFFSET=1): per DAT
# layer, CE between the softmax over the 20x20 cells and the uniform
# distribution over the cells inside the target window; every cell gets a
# gradient. Mini check (8 GPUs, ~45 min, same as miniB-qk + this):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 GRAD_ACCUM=1 \
#   DATA_JSON=$OSS_DATA/extra_0916/viscot_bbox.train.json MODEL_PATH=<v2-merged> \
#   TF_PROB=1 TF_FORCE_PROB=0.5 OFF_SUP_WEIGHT=10 REL_SUP_WEIGHT=1 LR_DROP_PROB=0.5 ROUTE_BY_DROP=1 \
#   OFF_HEAD_TRUNK_GRAD=0 GLOBAL_OFFSET=1 GLOB_REL=qk FREEZE_BASE=True LORA_ENABLE=False \
#   TUNE_MM_MLP=False MAX_STEPS=600 WARMUP_STEPS=20 SAVE_STEPS=300 EXP_NAME=0922_miniC_relsup bash <this script>
# wandb: dat/rel_sup_mass must climb well above dat/rel_sup_base (uniform map),
# dat/glob_scale must keep falling past 0.85 toward ~0.5, glob_shift past 0.15.
# Pass on the probe: r_cx/r_cy > 0.5, in_box well off 0.035.
# miniC result: the map took the SHAPE of the target (peak 40x -> 5x = a
# window-sized plateau) but not its place: window mass 0.33 of which 0.28 is
# a constant prior; r_cx 0.2, in_box 0.031, real 69.0 ~ shuffle 69.5.
#
# Diagnosis (0922 probe, per-head trunk attention on the held-out split):
# the intention token is the assistant <|im_start|> -- a FORMAT token; its
# attention over the LR cells has no question information (window mass
# 0.18-0.24 < 0.236 uniform, 0/8 heads). The token right before the answer
# ('\n' after 'assistant' = last prompt token) DOES: de-sinked 8-head mean
# mass 0.43-0.60 at layers 7-23, 8/8 heads above uniform, peak on the answer
# text (contact sheets: probe --dump_maps). Both 'qk' and 'xattn' rebuilt a
# text->cell matcher from scratch on top of a token that knows nothing, while
# the trunk's own q_proj/k_proj already had it. Fix = ask the right token and
# reuse the trunk's attention:
#   GLOB_QPOS=ans_prev   query token = the token before the answer
#   GLOB_REL=attn        map = the layer's own attention (heads averaged),
#                        sink cells masked (running mean > GLOB_SINK_X/N);
#                        parameter-free (0923: the learnable tau / cell bias
#                        were removed -- miniD: tau drifted 1 -> 0.974, the
#                        bias learned a dataset location prior, the raw
#                        de-sinked attention scored the same or better)
# Mini D result (0922, both arms 600 steps, held-out 500):
#   map:   attn mass 0.61 / argmax 0.70 / box 0.165 (miniB-qk 0.42 / - / 0.07)
#          -> passed; qk+ans_prev 0.42 / 0.69 / 0.11.
#   c:     r_cx 0.6 r_cy 0.77 both arms (miniB 0.1 / 0.45) -> ans_prev did it.
#   s:     stuck at 0.80-0.93, r_s ~ 0: whole-map moments are inflated by the
#          ~40% flat background; in_box 0.035 (unchanged), readout +0.8.
#   By outcome (probe 'by localisation outcome'): 73% of samples have the peak
#   in the window; on those real +1.9 / shuffle 0 / oracle +3.8; on the rest
#   real -2.2 and ORACLE 0 (label noise / unanswerable) -> localisation rate is
#   near its ceiling, the missing gain is on the hits. Every inference-only
#   grid swap (floor, argmax, gate) made the hits WORSE (+0.8..1.1): the
#   readout only reads the grid distribution it was trained with -> must
#   co-train (mini E). Head pooling (mean/max/best/conf/lse), cross-layer
#   fusion and sink thresholds 5/3/2 changed nothing.
#
# Full 0920 combined run (data = 0817 mix + viscot doc boxes + synth_hd text on
# SA-1B natural images (+ optional Visual-CoT natural-image boxes), built by
# scripts/compose_sft_mix.py):
#   DATA_JSON=$OSS_DATA/llava_hr_gen_vs_0817_bbox0920.json TF_PROB=1 TF_FORCE_PROB=0.5 \
#   OFF_SUP_WEIGHT=10 REL_SUP_WEIGHT=1 LR_DROP_PROB=0.5 ROUTE_BY_DROP=1 OFF_HEAD_TRUNK_GRAD=0 \
#   GLOBAL_OFFSET=1 GLOB_REL=qk \
#   EXP_NAME=0920_sft_qwen35_2b_dat_readers_bias_readout_qk bash <this script>
#
# Sanity: in the startup log check
#   [token-scheme] ... im_start=248045 (Qwen3.5 250k vocab resolved)
#   [patch-geometry] PATCH_SIZE=16 ... factor=32
#   trainable params include visual.merger.* (tune_mm_mlp) and lora_/dat keys

export WANDB_PROJECT="${WANDB_PROJECT:-vldat_experiments}"

export NUMEXPR_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------- Path config (OSS cluster) --------
OSS_DATA="${OSS_DATA:-/data/oss_bucket_0/wangziyi/models_data}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/pingping.wzy}"

CKPT_ROOT="${CKPT_ROOT:-$LOCAL_ROOT/vldat_experiments}"
MODEL_PATH="${MODEL_PATH:-$CKPT_ROOT/0915_pretrain_qwen35_2b_dat_readers_bias}"
CACHE_ROOT="${CACHE_ROOT:-$LOCAL_ROOT/cache/vldat}"
EXP_NAME="${EXP_NAME:-0915_sft_qwen35_2b_dat_readers_bias_genvs}"

# Exact merge gradients on for SFT too (stage-2 is where hd_kv keeps learning).
export DAT_EXACT_MERGE_GRAD=1
export DAT_ATTN_BACKEND="${DAT_ATTN_BACKEND:-auto}"

# train_split is a SYMLINK FARM on a LOCAL fs (OSS FUSE can't create symlinks).
IMAGE_ROOT="${IMAGE_ROOT:-$LOCAL_ROOT/sft_data}"

DATA_JSON="${DATA_JSON:-$OSS_DATA/llava_hr_gen_vs_0817.json}"

if [[ ! -f "$DATA_JSON" ]]; then
    echo "[ERROR] Missing $DATA_JSON (build via construct_sft_0817.py)" >&2; exit 1
fi
if [[ ! -d "$IMAGE_ROOT/train_split" ]]; then echo "[ERROR] Missing $IMAGE_ROOT/train_split (create it on LOCAL disk; OSS can't hold symlinks)" >&2; exit 1; fi
if [[ ! -e "$IMAGE_ROOT/train_split/sa1b" ]]; then echo "[ERROR] Missing sa1b symlink: ln -sfn $OSS_DATA/sa1b_images $IMAGE_ROOT/train_split/sa1b" >&2; exit 1; fi
# All prefixes referenced by the 0817 mix must be symlinked into the local farm.
for prefix in stvqa deepeyes visualprobe densefusion allava aokvqa scienceqa; do
    if [[ ! -e "$IMAGE_ROOT/train_split/$prefix" ]]; then
        echo "[ERROR] Missing $prefix symlink: ln -sfn $OSS_DATA/train_split/$prefix $IMAGE_ROOT/train_split/$prefix" >&2
        exit 1
    fi
done
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Missing stage-1 pretrain ckpt: $MODEL_PATH" >&2
    echo "        Run exp_pretrain_qwen35_2b_dat_readers_bias.sh first (or point MODEL_PATH at the OSS copy)." >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[ERROR] $MODEL_PATH lacks config.json (not a HF ckpt)" >&2; exit 1
fi

# -------- Preflight: env must actually support Qwen3.5 --------
python - <<'PY'
import sys
import transformers
print(f"[preflight] transformers {transformers.__version__}")
try:
    from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
except ImportError:
    sys.exit("[preflight ERROR] transformers has no Qwen3_5ForConditionalGeneration.\n"
             "  Qwen3.5 needs transformers >= 5.10; upgrade the env first.")
import flash_attn
print(f"[preflight] flash_attn {flash_attn.__version__}")
try:
    import fla
    print(f"[preflight] fla {getattr(fla, '__version__', '?')}")
except ImportError:
    print("[preflight WARN] fla (flash-linear-attention) missing — GDN layers use the slow torch fallback")
from llava.model.language_model import modeling_qwen3_5_dat as M
if not M._EXACT_MERGE_AVAILABLE:
    sys.exit(f"[preflight ERROR] exact merge gradients unavailable (backend={M._EXACT_BWD_BACKEND})")
print(f"[preflight] exact merge gradients via {M._EXACT_BWD_BACKEND.upper()} raw backward")
PY

mkdir -p "$CKPT_ROOT/$EXP_NAME"

# Compiled caches are PER HOST: the home is a shared JuiceFS, and a
# __triton_launcher.so built on a glibc-2.34 box fails to load on an older one.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton-$(hostname -s)}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor-$(hostname -s)}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

# -------- Single-node 8 GPU --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

# fla triton compile/autotune can stall ranks while others wait in allreduce.
# --ddp_timeout is IGNORED under deepspeed; DEEPSPEED_TIMEOUT (seconds) is
# what actually raises the process-group timeout there.
export DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-7200}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"

# MUST match stage 1 (dat_extra_args is rebuilt from CLI, not read from ckpt).
# Question readers stay ON (stage 1 trained with them). The LSE bias defaults
# to 0 here: the curriculum ran in stage 1 and the final stage-1 ckpt is
# unbiased; set HD_LSE_BIAS/HD_LSE_BIAS_DECAY for a short SFT-side curriculum.
DAT_LAYERS="${DAT_LAYERS:-auto}"

echo "[0915-sft] qwen3_5 2B  dat_layers=$DAT_LAYERS  grid=20  nogate  exact_grad=1  data=$(basename "$DATA_JSON")"

torchrun --nproc_per_node="${NPROC:-8}" --master_port "${MASTER_PORT:-40993}" llava/train/train_qwen_dat.py \
    --deepspeed ./scripts/zero_configs/zero2.json \
    --model_name_or_path "$MODEL_PATH" \
    --model_family qwen3_5 \
    --data_path "$DATA_JSON" \
    --image_folder "$IMAGE_ROOT/train_split" \
    --use_hr_first_resize False \
    --hd_max_pixels 5017600 \
    --use_dat True \
    --dat_layers "$DAT_LAYERS" \
    --dat_grid_size 20 \
    --dat_off_grps 8 \
    --dat_inter_size 128 \
    --dat_hr_scale 3 \
    --dat_hd_proj True \
    --dat_use_intention_branch True \
    --dat_intention_as_gate True \
    --dat_use_spatial_attn_guide False \
    --dat_shared_vit False \
    --dat_freeze_base "${FREEZE_BASE:-False}" \
    --dat_warmup_steps 0 \
    --dat_inject_lr_image False \
    --dat_off_penalty "${OFF_PENALTY:-1.0}" \
    --dat_off_range "${OFF_RANGE:-0}" \
    --dat_intention_inject "${INTENTION_INJECT:-gate}" \
    --dat_image_hd_for_question "${QUESTION_HD:-True}" \
    --dat_hd_lse_bias "${HD_LSE_BIAS:-0}" \
    --dat_hd_lse_bias_decay_steps "${HD_LSE_BIAS_DECAY:-0}" \
    --dat_tf_prob "${TF_PROB:-0}" \
    --dat_tf_min_cells "${TF_MIN_CELLS:-20}" \
    --dat_tf_min_hd_ratio "${TF_MIN_HD_RATIO:-2.0}" \
    --dat_tf_max_window_frac "${TF_MAX_WINDOW_FRAC:-0.5}" \
    --dat_lr_drop_prob "${LR_DROP_PROB:-0}" \
    --dat_lr_drop_ratio "${LR_DROP_RATIO:-0.75}" \
    --dat_off_sup_weight "${OFF_SUP_WEIGHT:-0}" \
    --dat_off_sup_delta "${OFF_SUP_DELTA:-0.1}" \
    --dat_tf_force_prob "${TF_FORCE_PROB:--1}" \
    --dat_route_by_lr_drop "$([ "${ROUTE_BY_DROP:-0}" = 1 ] && echo True || echo False)" \
    --dat_off_head_trunk_grad "${OFF_HEAD_TRUNK_GRAD:-1.0}" \
    --dat_use_global_offset "$([ "${GLOBAL_OFFSET:-0}" = 1 ] && echo True || echo False)" \
    --dat_glob_min_scale "${GLOB_MIN_SCALE:-0.1}" \
    --dat_glob_relevance "${GLOB_REL:-conv}" \
    --dat_glob_dim "${GLOB_DIM:-128}" \
    --dat_glob_query_pos "${GLOB_QPOS:-ans_prev}" \
    --dat_glob_sink_x "${GLOB_SINK_X:-5}" \
    --dat_rel_sup_weight "${REL_SUP_WEIGHT:-0}" \
    --dat_lr 1e-4 \
    --lora_enable "${LORA_ENABLE:-True}" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_layers "all" \
    --lora_lr 2e-5 \
    --tune_mm_vision False \
    --tune_mm_mlp "${TUNE_MM_MLP:-True}" \
    --tune_mm_llm False \
    --mm_projector_lr 5e-6 \
    --kd_on False \
    --bf16 True \
    --tf32 True \
    --max_grad_norm 1.0 \
    --ddp_timeout 7200 \
    --output_dir "$CKPT_ROOT/$EXP_NAME" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --max_steps "${MAX_STEPS:--1}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH:-4}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-2}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-500}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps "${WARMUP_STEPS:-50}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --group_by_modality_length True \
    --dataloader_num_workers "${DATALOADER_WORKERS:-8}" \
    --dataloader_pin_memory True \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --dataloader_drop_last True \
    --seed 42 \
    --report_to "wandb" \
    --run_name "$EXP_NAME"

# Auto-merge LoRA + non-LoRA trainables (DAT params + projector deltas)
# into a self-contained HF ckpt at $CKPT_ROOT/$EXP_NAME-merged.
source "$(dirname "${BASH_SOURCE[0]}")/../qwen2_5vl_adl_0701/_merge_after_train.sh"
