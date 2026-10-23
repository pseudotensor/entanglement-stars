#!/bin/bash
# run_v3_rd1c.sh (busi, analyst, 21 Oct): RD-1c's stored arms re-read by tolcmp_v3 (ReviewA_rd1c_v1 sec. 2 (1)), box CPU, nice 10, 8 threads.
# card: X = B7, D = A6 (h100f/gate), P = the R2 card's v9c3 segment; box smoke: X = H12, D = T12 (seg/), same P. Also a v2 rerun of the card
# read as a reproduction check of the stored tolcmp_card.txt (1d2aa821).
set -u
R=${QI2_ROOT}/RD1c_build; V=$R/tolcmp_v3; PY=${QI2_ROOT}/python
P=${QI2_ROOT}/segment_c64s8L_lagged_sector_v9c3_anchor_passage_lz_20261020T091433Z
T=C64_s8_ell8_tau2_twofield_v4L
E="env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10"
cd $R
$E $PY -I $V/code/tolcmp_v3.py h100f/gate/RD1CAB_B7_20261021T175449Z h100f/gate/RD1CAB_A6_20261021T174101Z $P --tag $T --after 13 --rs 64 --names B7,A6,P_R2 --out $V/card > $V/tolcmp_v3_card.txt 2>&1; echo "card v3 rc $?"
$E $PY -I $V/code/tolcmp_v3.py seg/H12 seg/T12 $P --tag $T --after 13 --rs 64 --names H12,T12,P_R2 --out $V/box > $V/tolcmp_v3_box.txt 2>&1; echo "box v3 rc $?"
$E $PY -I $V/code/tolcmp_v2.py h100f/gate/RD1CAB_B7_20261021T175449Z h100f/gate/RD1CAB_A6_20261021T174101Z $P --tag $T --after 13 --rs 64 --names B7,A6,P_R2 --out $V/v2rerun_card > $V/tolcmp_v2rerun_card.txt 2>&1; echo "card v2 rerun rc $?"
