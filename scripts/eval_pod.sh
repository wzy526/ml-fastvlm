#!/usr/bin/env bash
# Thin wrapper: load pod offline env, then run the pixel sweep.
# Usage: bash scripts/eval_pod.sh dat35 <CKPT> <TASK> [TAG]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/eval_pod_env.sh"
exec bash "$SCRIPT_DIR/eval_pixel_sweep.sh" "$@"
