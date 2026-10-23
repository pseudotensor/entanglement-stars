#!/bin/bash
# run_rd2_ruling_v2.sh (busi, analyst, 21 Oct): ruling v2 (deliveries/lead_gaps/ReviewA_tolv2_v1.md) re-read of the stored RD-2 / RD-2b arms with
# cmp_rd2_ruling_v2.py, and the sec. 4 (a) scope audit with cmp_rd2_scope_audit_v1.py. Box CPU, nice 10, 8 threads; reads only; no card time.
# Segments: RD-2b S2 (I, C), RD-2 S2 (I, C, W), RD-2 S1 (I, C, W), the RD-2 box replay (I, C, W).
# Outputs: ${QI2_ROOT}/RD2_ruling_v2 (cmp_*_v2.{txt,json}, audit_*.{txt,json}, MD5SUMS). The v1 outputs (incl. the
# S2 firstscope file) stay where they are, untouched.
set -u
PY=${QI2_ROOT}/python
B=${QI2_ROOT}/perf_redesign; O=$B/RD2_ruling_v2; C=$O/code
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd $O
P2=${QI2_ROOT}/RUNG2c40_v36q
P1=${QI2_ROOT}/T0012.00
SB=$B/RD2b_build/h100f/S2; S2=$B/RD2_build/h100f/v2/S2; S1=$B/RD2_build/h100f/v2/S1; R=$B/RD2_build/replay
V2="nice -n 10 $PY -B -I $C/cmp_rd2_ruling_v2.py --shared $C/tolcmp_v2.py"
AU="nice -n 10 $PY -B -I $C/cmp_rd2_scope_audit_v1.py"
EXTRA=${EXTRA:-}
X=""; [ -n "$EXTRA" ] && X="--gate-extra $EXTRA"
SUF=${SUF:-}
JOBS=${1:-"rd2b_S2 rd2_S2 rd2_S1 rd2_box audit"}
for j in $JOBS; do
  case $j in
    rd2b_S2) $V2 $X --label "RD-2b S2 (h100-f, v36nq_rd2 on v16e)" --arm I=$SB/I:RUNG2c40_v36q_rd2bI --arm C=$SB/C:RUNG2c40_v36q_rd2bC --rs 64 \
               --prod-json $P2/collapse_RUNG2c40_v36q.json --prod-dir $P2 --prod-tag RUNG2c40_v36q --t-min 2.0 \
               --slice I=$SB/I/states/orbitals_RUNG2c40_v36q_rd2bI_T0003.00.npz --slice C=$SB/C/states/orbitals_RUNG2c40_v36q_rd2bC_T0003.00.npz \
               --percall C=gate:$B/RD2b_build/gate/rd2b_field_gate_s2replay.json --out cmp_rd2b_S2_IC_v2$SUF.json > cmp_rd2b_S2_IC_v2$SUF.txt 2>&1;;
    rd2_S2) $V2 $X --label "RD-2 S2 (h100-f, v36nq_rd2 on v16d)" --arm I=$S2/I:RUNG2c40_v36q_rd2I --arm C=$S2/C:RUNG2c40_v36q_rd2C --arm W=$S2/W:RUNG2c40_v36q_rd2W --rs 64 \
               --prod-json $P2/collapse_RUNG2c40_v36q.json --prod-dir $P2 --prod-tag RUNG2c40_v36q --t-min 2.0 \
               --slice I=$S2/I/states/orbitals_RUNG2c40_v36q_rd2I_T0003.00.npz --slice C=$S2/C/states/orbitals_RUNG2c40_v36q_rd2C_T0003.00.npz \
               --slice W=$S2/W/states/orbitals_RUNG2c40_v36q_rd2W_T0003.00.npz \
               --percall C=route-identity --percall W=route-identity --out cmp_rd2_S2_ICW_v2$SUF.json > cmp_rd2_S2_ICW_v2$SUF.txt 2>&1;;
    rd2_S1) $V2 $X --label "RD-2 S1 (h100-f, v37a_rd2 on v16d)" --arm I=$S1/I:RUNG2c40b_srf_rd2I --arm C=$S1/C:RUNG2c40b_srf_rd2C --arm W=$S1/W:RUNG2c40b_srf_rd2W --rs 64 \
               --prod-json $P1/collapse_RUNG2c40b_srf.json --prod-dir $P1 --prod-tag RUNG2c40b_srf --t-min 12.0 \
               --slice I=$S1/I/states/orbitals_RUNG2c40b_srf_rd2I_T0012.50.npz --slice C=$S1/C/states/orbitals_RUNG2c40b_srf_rd2C_T0012.50.npz \
               --slice W=$S1/W/states/orbitals_RUNG2c40b_srf_rd2W_T0012.50.npz --out cmp_rd2_S1_ICW_v2$SUF.json > cmp_rd2_S1_ICW_v2$SUF.txt 2>&1;;
    rd2_box) $V2 $X --label "RD-2 box replay (box CPU, v37a_rd2 on v16d)" --arm I=$R/I:RUNG2c40b_srf_rd2I --arm C=$R/C:RUNG2c40b_srf_rd2C --arm W=$R/W:RUNG2c40b_srf_rd2W --rs 64 \
               --prod-json $P1/collapse_RUNG2c40b_srf.json --prod-dir $P1 --prod-tag RUNG2c40b_srf --t-min 12.0 --out cmp_rd2_box_ICW_v2$SUF.json > cmp_rd2_box_ICW_v2$SUF.txt 2>&1;;
    audit) $AU --label "RD-2 S2" --arm I=$S2/I:RUNG2c40_v36q_rd2I --arm C=$S2/C:RUNG2c40_v36q_rd2C --arm W=$S2/W:RUNG2c40_v36q_rd2W --prod-json $P2/collapse_RUNG2c40_v36q.json --t-min 2.0 --out audit_rd2_S2.json > audit_rd2_S2.txt 2>&1
           $AU --label "RD-2 S1" --arm I=$S1/I:RUNG2c40b_srf_rd2I --arm C=$S1/C:RUNG2c40b_srf_rd2C --arm W=$S1/W:RUNG2c40b_srf_rd2W --prod-json $P1/collapse_RUNG2c40b_srf.json --t-min 12.0 --out audit_rd2_S1.json > audit_rd2_S1.txt 2>&1
           $AU --label "RD-2 box replay" --arm I=$R/I:RUNG2c40b_srf_rd2I --arm C=$R/C:RUNG2c40b_srf_rd2C --arm W=$R/W:RUNG2c40b_srf_rd2W --prod-json $P1/collapse_RUNG2c40b_srf.json --t-min 12.0 --out audit_rd2_box.json > audit_rd2_box.txt 2>&1
           $AU --label "RD-2b S2 (beside the audit)" --arm I=$SB/I:RUNG2c40_v36q_rd2bI --arm C=$SB/C:RUNG2c40_v36q_rd2bC --prod-json $P2/collapse_RUNG2c40_v36q.json --t-min 2.0 --out audit_rd2b_S2.json > audit_rd2b_S2.txt 2>&1;;
    *) echo "unknown job $j"; exit 2;;
  esac
  echo "$j rc $? $(date -u +%FT%TZ)"
done
