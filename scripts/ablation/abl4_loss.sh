#!/usr/bin/env bash
# Ablation 4, density-head losses. Final recipe: answer CE 1, box KL 0.1, trust
# region 0.1, hinge 1, no density-ratio cap (comp_qlas of ablation 1).
#   Earlier heads, same where-head / data / steps, trained under a ratio cap of 4,
#   so they are scored with the cap: hinge, centre L2, both. Eval only.
#   (stage1_ldat.pt, cap 4 without hinge, scored 543 / 800 on HR-Bench 4K.)
#   New heads: final recipe without hinge / box KL / trust region. 3 trainings.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need "$QA"

for v in "r4_hinge:hinge_ldat" "r4_ctr:ctrl2_ldat" "r4_hinge_ctr:both_ldat"; do
    f=$X/${v#*:}.pt
    if [ -s "$f" ]; then
        ev "loss_${v%%:*}" LDAT_PT="$f" DAT_RATIO=4
    else
        say "skip loss_${v%%:*}: $f missing"
    fi
done

for v in "no_hinge:--ans_w 1 --box_w 0.1 --tr_w 0.1 --hinge_w 0" \
         "no_box:--ans_w 1 --box_w 0 --tr_w 0.1 --hinge_w 1" \
         "no_tr:--ans_w 1 --box_w 0.1 --tr_w 0 --hinge_w 1"; do
    n=loss_${v%%:*}
    train_ldat "$n" "$(train_base "$M2" "$W2")" "${v#*:}" && ev "$n" LDAT_PT="$A/heads/$n.pt"
done
say "done"
