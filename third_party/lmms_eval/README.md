# lmms-eval DAT model wrapper

`qwen2_5_dat_vl.py` is the lmms-eval model class used for every DAT evaluation in
this repo (`--model qwen2_5_dat_vl`, see `scripts/eval_pixel_sweep.sh`). It lives
in the lmms-eval fork at `lmms_eval/models/simple/qwen2_5_dat_vl.py`; that fork is
not under version control, so the file is mirrored here and this copy is the
reference. To install, copy it over the fork's file.

Changes carried by this copy (2026-09-06/07):

- Geometry fix in `_derive_hr_size_from_lr_first`: `image_grid_thw` counts
  unmerged patches, so LR pixels scale with `patch_size` (16), not `_factor`
  (32). The old code over-estimated LR area 4x, made `hd_target` 36x the LR area
  instead of `hr_scale^2` = 9x, and pinned every sweep point to `hr_max_pixels`
  (the 0903 campaign). Every 0906+ number uses the fixed geometry.
- HR size is passed to `fetch_image` as `resized_height/width` instead of
  `min_pixels=max_pixels`, avoiding `height and width must be > 0` on extreme
  aspect ratios; `generate_until` degrades a failing sample to an empty answer
  instead of deadlocking the other ranks.
- `disable_hd=True` model arg (`DISABLE_HD=1` in `eval_pixel_sweep.sh`): runs the
  DAT checkpoint with the HD branch off, i.e. the SFT-only control.
- `_FASTVLM_PATH` resolution so `llava` imports from env / relative / CFS paths.
