#!/usr/bin/env bash
# =============================================================================
# download_all_data.sh — one-shot dataset provisioning for a fresh cluster.
#
# Target layout (everything under $DATA_BASE, meant to sit on the OSS mount):
#   $DATA_BASE/models_data/
#     ├── sft_data/
#     │   ├── train_split/{coco,vg,gqa,ocr_vqa,textvqa,      <- from LLaVA-665K
#     │   │                 synthdog,docvqa,chartqa,infovqa,  <- from construct_hd251k
#     │   │                 ai2d,sam}
#     │   │   └── sa1b -> ../../sa1b_images   (symlink, created by builder)
#     │   ├── llava_v1_5_mix665k*.json        <- from LLaVA-665K
#     │   ├── llava_hd251k.json               <- built by construct_hd251k.py
#     │   ├── llava_hd_merged_1m.json         <- built by merge_datasets.py (see notes)
#     │   ├── llava_sa1b_caption_pretrain.json     <- PRETRAIN data (built here)
#     │   └── llava_hr_essential_sa1b_ivcap.json   <- SFT data     (built here)
#     ├── sa1b_images/sa_NNNNNN/*.jpg         <- from download_sa1b_shards.sh
#     ├── InternVL-SA-1B-Caption/*.jsonl      <- from OpenGVLab/InternVL-SA-1B-Caption
#     └── sa1b_links.txt                      <- (optional) Meta presigned links
#
# Usage (defaults are set for the new cluster):
#   bash scripts/download_all_data.sh                 # run everything, resumable
#   ONLY=llava665k bash scripts/download_all_data.sh  # run a single step
#   SA1B_END_SHARD=44 bash scripts/download_all_data.sh
#
# Resumable: each step writes a marker in $DATA_BASE/.download_markers/.
# Re-running skips completed steps. Delete a marker to force redo.
#
# NOT `set -e`: a single failing step should not abort the rest.
# =============================================================================
set -uo pipefail

# -------- paths (override via env) --------
REPO="${REPO:-/home/pingping.wzy/ml-fastvlm}"
DATA_BASE="${DATA_BASE:-/data/oss_bucket_0/wangziyi}"
MODELS_DATA="$DATA_BASE/models_data"
SFT_DATA="$MODELS_DATA/sft_data"
TRAIN_SPLIT="$SFT_DATA/train_split"
SA1B_DIR="$MODELS_DATA/sa1b_images"
INTERNVL_DIR="$MODELS_DATA/InternVL-SA-1B-Caption"
# Meta presigned links for SA-1B (valid until ~2026-08-01). Lives in the repo
# so it ships with the code; override with LINKS_FILE=/path if you refresh it.
LINKS_FILE="${LINKS_FILE:-$REPO/sa1b_links_0803.txt}"
MARK_DIR="$DATA_BASE/.download_markers"
TMP_DIR="${TMP_DIR:-$DATA_BASE/_dl_tmp}"

# SA-1B shard range (0..44 = 45 shards ≈ 465 GB, matches the reference machine).
SA1B_START_SHARD="${SA1B_START_SHARD:-0}"
SA1B_END_SHARD="${SA1B_END_SHARD:-44}"

# HF mirror (unset HF_ENDPOINT before running to hit huggingface.co directly).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Optional single-step selector: llava665k | internvl | sa1b | hd251k | pretrain_json | sft_json
ONLY="${ONLY:-}"

CONDA_ENV="${CONDA_ENV:-vldat}"

# -------- helpers --------
_c_grn='\033[0;32m'; _c_red='\033[0;31m'; _c_yel='\033[0;33m'; _c_off='\033[0m'
log()  { echo -e "${_c_grn}[dl]$( date +%H:%M:%S ) $*${_c_off}"; }
warn() { echo -e "${_c_yel}[dl][WARN] $*${_c_off}"; }
err()  { echo -e "${_c_red}[dl][ERR ] $*${_c_off}" >&2; }
done_ok() { [[ -f "$MARK_DIR/$1.done" ]]; }
mark()    { mkdir -p "$MARK_DIR"; touch "$MARK_DIR/$1.done"; }
want()    { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

mkdir -p "$MODELS_DATA" "$SFT_DATA" "$TRAIN_SPLIT" "$SA1B_DIR" "$INTERNVL_DIR" "$MARK_DIR" "$TMP_DIR"

# -------- conda + tools --------
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" && conda activate "$CONDA_ENV" 2>/dev/null || \
        warn "conda env '$CONDA_ENV' not activated; using current python."
fi
HF_CLI=""
if command -v huggingface-cli >/dev/null 2>&1; then HF_CLI="huggingface-cli";
elif command -v hf >/dev/null 2>&1; then HF_CLI="hf"; fi
[[ -z "$HF_CLI" ]] && err "huggingface-cli / hf not found. pip install -U huggingface_hub"

log "REPO=$REPO"
log "DATA_BASE=$DATA_BASE   HF_ENDPOINT=$HF_ENDPOINT"
log "SA-1B shards: sa_$(printf '%06d' "$SA1B_START_SHARD")..sa_$(printf '%06d' "$SA1B_END_SHARD")"
echo

# =============================================================================
# STEP 1: LLaVA-1.5-665K bundle  (coco / vg / gqa / ocr_vqa / textvqa + jsons)
#   repo: kaiyuyue/llava-1.5-665k-instructions  (~77 GB)
# =============================================================================
if want llava665k && ! done_ok llava665k; then
    log "STEP 1/6  LLaVA-1.5-665K bundle -> $SFT_DATA"
    if [[ -z "$HF_CLI" ]]; then
        err "  STEP 1 skipped: no huggingface CLI. Run: pip install -U 'huggingface_hub[cli]', then re-run."
    elif "$HF_CLI" download --repo-type dataset kaiyuyue/llava-1.5-665k-instructions \
            --local-dir "$SFT_DATA" ; then
        log "  extracting train_split/*.tar.gz ..."
        shopt -s nullglob
        for tarf in "$SFT_DATA"/train_split/*.tar.gz "$SFT_DATA"/train_split/*.tar; do
            name="$(basename "$tarf")"; name="${name%%.tar*}"
            if [[ -d "$TRAIN_SPLIT/$name" ]] && [[ -n "$(ls -A "$TRAIN_SPLIT/$name" 2>/dev/null)" ]]; then
                log "    $name already extracted, skip"; continue
            fi
            log "    extracting $name ..."
            tar -xf "$tarf" -C "$TRAIN_SPLIT" && rm -f "$tarf"
        done
        shopt -u nullglob
        mark llava665k
        log "  STEP 1 done."
    else
        err "  STEP 1 FAILED (huggingface download). Re-run to resume."
    fi
    echo
fi

# =============================================================================
# STEP 2: InternVL SA-1B captions  (pretrain text; en single-image jsonl only)
#   repo: OpenGVLab/InternVL-SA-1B-Caption
# =============================================================================
if want internvl && ! done_ok internvl; then
    log "STEP 2/6  InternVL-SA-1B-Caption (en 11M jsonl) -> $INTERNVL_DIR"
    if [[ -z "$HF_CLI" ]]; then
        err "  STEP 2 skipped: no huggingface CLI. Run: pip install -U 'huggingface_hub[cli]', then re-run."
    elif "$HF_CLI" download --repo-type dataset OpenGVLab/InternVL-SA-1B-Caption \
            internvl_sa1b_caption_11m_single_image_en.jsonl \
            --local-dir "$INTERNVL_DIR" ; then
        mark internvl
        log "  STEP 2 done."
    else
        err "  STEP 2 FAILED. Re-run to resume."
    fi
    echo
fi

# =============================================================================
# STEP 3: SA-1B image shards  (pretrain images + sft sa1b subset)
#   Uses the repo's downloader forced to the Meta presigned-links channel
#   ($LINKS_FILE). The HF mirror (sailvideo/SA-1B) returns 404, so meta only.
#   NOTE: the links in sa1b_links_0803.txt expire ~2026-08-01 — refresh from
#   https://ai.meta.com/datasets/segment-anything-downloads/ if they lapse.
# =============================================================================
if want sa1b && ! done_ok sa1b; then
    log "STEP 3/6  SA-1B shards -> $SA1B_DIR  (SOURCE=meta, links=$LINKS_FILE)"
    if [[ ! -f "$LINKS_FILE" ]]; then
        err "  LINKS_FILE not found: $LINKS_FILE"
        err "  -> copy sa1b_links_0803.txt into the repo, or set LINKS_FILE=/path, then re-run."
    elif SA1B_DIR="$SA1B_DIR" LINKS_FILE="$LINKS_FILE" TMP_DIR="$TMP_DIR/sa1b" SOURCE=meta \
       START_SHARD="$SA1B_START_SHARD" END_SHARD="$SA1B_END_SHARD" \
       bash "$REPO/scripts/qwen2_5vl_adl_0430/download_sa1b_shards.sh" ; then
        mark sa1b
        log "  STEP 3 done."
    else
        err "  STEP 3 FAILED / partial. Re-run to resume (already-extracted shards are skipped)."
    fi
    echo
fi

# =============================================================================
# STEP 4: HD-251K SFT datasets  (synthdog/docvqa/chartqa/infovqa/textvqa/sam/ai2d)
#   downloads from HF and writes train_split/<set>/ + llava_hd251k.json
# =============================================================================
if want hd251k && ! done_ok hd251k; then
    log "STEP 4/6  construct_hd251k.py -> $SFT_DATA (train_split + llava_hd251k.json)"
    if ( cd "$REPO" && python construct_hd251k.py --sft_dir "$SFT_DATA" --hf_mirror ) ; then
        mark hd251k
        log "  STEP 4 done."
    else
        err "  STEP 4 FAILED. Re-run to resume (per-dataset caches are reused)."
    fi
    echo
fi

# =============================================================================
# STEP 5: build PRETRAIN json  (llava_sa1b_caption_pretrain.json)
#   needs: SA-1B images (step 3) + InternVL captions (step 2)
# =============================================================================
if want pretrain_json && ! done_ok pretrain_json; then
    log "STEP 5/6  build pretrain json"
    if ( cd "$REPO" && python scripts/qwen2_5vl_adl_0430/build_sa1b_caption_pretrain.py \
            --sa1b_image_dir "$SA1B_DIR" \
            --internvl_dir   "$INTERNVL_DIR" \
            --output_json    "$SFT_DATA/llava_sa1b_caption_pretrain.json" \
            --data_root      "$SFT_DATA" ) ; then
        mark pretrain_json
        log "  STEP 5 done -> $SFT_DATA/llava_sa1b_caption_pretrain.json"
    else
        err "  STEP 5 FAILED."
    fi
    echo
fi

# =============================================================================
# STEP 6: build SFT json  (llava_hr_essential_sa1b_ivcap.json)
#   chain: merge_datasets.py -> build_hr_essential_mix.py -> build_sa1b_mix.py
#
#   NOTE — llava_hd_merged_1m.json provenance:
#   It is produced by merge_datasets.py from the LLaVA-665K
#   'llava_v1_5_mix665k_shuffled.json' (step 1) plus a 'hd_supplements.jsonl'.
#   If hd_supplements.jsonl is not present we CANNOT regenerate it here; in that
#   case copy llava_hd_merged_1m.json (or the final SFT json) from the old host:
#     scp <old>:/root/autodl-tmp/models_data/sft_data/llava_hr_essential_sa1b_ivcap.json \
#         $SFT_DATA/
#   The two final training JSONs are small (~0.3-0.5 GB) and safe to scp.
# =============================================================================
if want sft_json && ! done_ok sft_json; then
    log "STEP 6/6  build SFT json"
    FINAL_SFT="$SFT_DATA/llava_hr_essential_sa1b_ivcap.json"
    HD_MERGED="$SFT_DATA/llava_hd_merged_1m.json"
    HR350="$SFT_DATA/llava_hr_essential_350k.json"

    if [[ -f "$FINAL_SFT" ]]; then
        log "  $FINAL_SFT already present (copied?), marking done."
        mark sft_json
    else
        # 6a. ensure llava_hd_merged_1m.json
        if [[ ! -f "$HD_MERGED" ]]; then
            warn "  $HD_MERGED missing — trying merge_datasets.py"
            if [[ -f "$SFT_DATA/llava_v1_5_mix665k_shuffled.json" && -f "$SFT_DATA/hd_supplements.jsonl" ]]; then
                ( cd "$REPO" && SFT_DATA_DIR="$SFT_DATA" python merge_datasets.py ) || \
                    err "  merge_datasets.py failed."
            else
                err "  Cannot build $HD_MERGED (missing llava_v1_5_mix665k_shuffled.json or hd_supplements.jsonl)."
                err "  -> scp llava_hd_merged_1m.json OR the final llava_hr_essential_sa1b_ivcap.json from the old host, then re-run."
            fi
        fi
        # 6b. hr_essential_350k
        if [[ -f "$HD_MERGED" && ! -f "$HR350" ]]; then
            ( cd "$REPO" && python scripts/qwen2_5vl_adl_0430/build_hr_essential_mix.py \
                --source_json  "$HD_MERGED" \
                --hd251k_json  "$SFT_DATA/llava_hd251k.json" \
                --output_json  "$HR350" \
                --image_folder "$TRAIN_SPLIT" ) || err "  build_hr_essential_mix.py failed."
        fi
        # 6c. add sa1b + ivcap -> final
        if [[ -f "$HR350" ]]; then
            ( cd "$REPO" && python scripts/qwen2_5vl_adl_0430/build_sa1b_mix.py \
                --sa1b_image_dir    "$SA1B_DIR" \
                --internvl_dir      "$INTERNVL_DIR" \
                --hr_essential_json "$HR350" \
                --output_json       "$FINAL_SFT" \
                --no_as_core ) && mark sft_json || err "  build_sa1b_mix.py failed."
        fi
    fi
    [[ -f "$FINAL_SFT" ]] && log "  STEP 6 done -> $FINAL_SFT"
    echo
fi

# =============================================================================
log "=========================== SUMMARY ==========================="
for s in llava665k internvl sa1b hd251k pretrain_json sft_json; do
    if done_ok "$s"; then echo -e "  ${_c_grn}[ok]${_c_off}   $s"; else echo -e "  ${_c_red}[todo]${_c_off} $s"; fi
done
echo
log "Pretrain data: $SFT_DATA/llava_sa1b_caption_pretrain.json"
log "SFT data:      $SFT_DATA/llava_hr_essential_sa1b_ivcap.json"
log "Images root:   $TRAIN_SPLIT   (sa1b -> $SA1B_DIR)"
log "Done. Re-run to resume any [todo] step."
