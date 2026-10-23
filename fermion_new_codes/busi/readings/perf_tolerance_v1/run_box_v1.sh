#!/bin/bash
# run_box_v1.sh (reviewer A perf tolerance ruling v1): the stored replays compared by quantity class, on the box CPU (nice 10, 8 threads).
# Outputs: ${QI2_ROOT}/{symc_*,symw1_*}.{txt,json}
set -u
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
OUT=${QI2_ROOT}/perf_tolerance_v1
CODE=$OUT/code
mkdir -p "$OUT"
cd "$OUT"
TC="nice -n 10 $PY $CODE/tolcmp_v1.py"

# --- SYM-C: twofield v9c5 (SYM = A1+C1+SYM-C1+SYM-C2) against v9c4 (AC = A1+C1), the h100-f card A/B
TAGC=C64_s8_ell8_tau2_twofield_v4L
SC=${QI2_ROOT}/h100f
H3=${QI2_ROOT}/gate
R2=${QI2_ROOT}/segment_c64s8L_lagged_sector_v9c3_anchor_passage_lz_20261020T091433Z
$TC $SC/SYMAB_SYM_20261021T055015Z $SC/SYMAB_AC_20261021T053101Z --tag $TAGC --after 13.0 --nameA "SYM(v9c5 C1+C2, h100-f)" --nameB "AC(v9c4, h100-f)" --out symc_SYM_vs_AC.json > symc_SYM_vs_AC.txt 2>&1
$TC $SC/SYMAB_AC_20261021T053101Z $H3/H3AB_AC_20261020T235625Z --tag $TAGC --after 13.0 --nameA "AC(h100-f 05:31)" --nameB "AC(h100-f 23:56, H3 A/B)" --out symc_AC_vs_H3AC.json > symc_AC_vs_H3AC.txt 2>&1
$TC $SC/SYMAB_AC_20261021T053101Z $R2 --tag $TAGC --after 13.0 --nameA "AC(v9c4, h100-f)" --nameB "v9c3 defaults (R2 card)" --out symc_AC_vs_R2.json > symc_AC_vs_R2.txt 2>&1
$TC $SC/SYMAB_SYM_20261021T055015Z $R2 --tag $TAGC --after 13.0 --nameA "SYM(v9c5, h100-f)" --nameB "v9c3 defaults (R2 card)" --out symc_SYM_vs_R2.json > symc_SYM_vs_R2.txt 2>&1
$TC $H3/H3AB_H3_0_20261020T233604Z $R2 --tag $TAGC --after 13.0 --nameA "H3_0(v9c3c, h100-f)" --nameB "v9c3 defaults (R2 card)" --out symc_H30_vs_R2.json > symc_H30_vs_R2.txt 2>&1

# --- SYM-W1: v36w2 chebyshev against eigh (h100-f replay of R3b T0038.00 -> 39.0), and each against the production (R3b H100)
TAGW=lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_v36v_anc
RP=${QI2_ROOT}/replay
PJ=${QI2_ROOT}/collapse_$TAGW.json
MIR=${QI2_ROOT}/T0039.00
SL=orbitals_${TAGW}_T0039.00.npz
$TC $RP/cheb $RP/eigh --tag $TAGW --after 38.0 --nameA "cheb(v36w2, h100-f)" --nameB "eigh(v36w2 default, h100-f)" --slice-a $RP/cheb/states/$SL --slice-b $RP/eigh/states/$SL --out symw1_cheb_vs_eigh.json > symw1_cheb_vs_eigh.txt 2>&1
$TC $RP/eigh $MIR --tag $TAGW --after 38.0 --nameA "eigh(h100-f)" --nameB "production (R3b H100)" --prod-json $PJ --prod-npz-dir $MIR --slice-a $RP/eigh/states/$SL --slice-b $MIR/$SL --out symw1_eigh_vs_prod.json > symw1_eigh_vs_prod.txt 2>&1
$TC $RP/cheb $MIR --tag $TAGW --after 38.0 --nameA "cheb(h100-f)" --nameB "production (R3b H100)" --prod-json $PJ --prod-npz-dir $MIR --slice-a $RP/cheb/states/$SL --slice-b $MIR/$SL --out symw1_cheb_vs_prod.json > symw1_cheb_vs_prod.txt 2>&1
md5sum *.txt *.json > MD5SUMS_perf_tolerance_v1.txt
echo DONE
