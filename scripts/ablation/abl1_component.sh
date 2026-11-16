#!/usr/bin/env bash
# Ablation 1, components. Same LR view and 400 extra tokens in every row; only
# how those tokens are taken changes. Eval only, 7 rows.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkzero "$Z2" "$M2"

# Window at the where-head peak, uniform resize (alpha 0 = uniform density).
ev comp_crop_fixed DAT_ALPHA=0 DAT_K=0
ev comp_crop_adapt DAT_ALPHA=0
# Warp inside the adaptive window: where-head density only, then + learned residual.
ev comp_dat_zero LDAT_PT="$Z2"
ev comp_qlas
# No window: the whole image, uniform / where-head warp / learned warp.
ev comp_full_uniform WIN_FULL=1 DAT_ALPHA=0
ev comp_full_dat WIN_FULL=1 LDAT_PT="$Z2"
ev comp_full_qlas WIN_FULL=1
say "done"
