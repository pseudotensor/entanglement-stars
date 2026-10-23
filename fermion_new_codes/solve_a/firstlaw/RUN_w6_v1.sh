#!/bin/bash
# RUN_w6_v1.sh: W6_FIRSTLAW_rungs_v1 from the v4b outputs on the box (CPU, minutes)
set -e
C=${QI2_ROOT}/qi2_w6_code; F=${QI2_ROOT}/firstlaw; O=$F/w6_rungs_v1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=4
mkdir -p $O; cd $C
T2=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d; T4=c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40; T8=c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40
nice -n 10 $PY w6_readout_v1.py --ladders ${QI2_ROOT}/ladders_readout_v1_1.json --phases 14.75,17.5 --out-dir $O \
  --rung s2_engine=2=$F/sigma2_point_v4b_ek/firstlaw_records_v4b_$T2.json --rung s4_engine=4=$F/sigma4_rung_v4b_ek/firstlaw_records_v4b_$T4.json \
  --rung s8_engine=8=$F/sigma8_rung_v4b_ek/firstlaw_records_v4b_$T8.json \
  --rung s2_analytic=2=$F/sigma2_point_v4b/firstlaw_records_v4b_$T2.json --rung s4_analytic=4=$F/sigma4_rung_v4b/firstlaw_records_v4b_$T4.json \
  --rung s8_analytic=8=$F/sigma8_rung_v4b/firstlaw_records_v4b_$T8.json
nice -n 10 $PY w6_notes_v1.py --readout $O/w6_readout_v1.json --code-dir $C --out $O/w6_notes_v1.json
nice -n 10 $PY w6_compose_v1.py --readout $O/w6_readout_v1.json --notes $O/w6_notes_v1.json --out $O/W6_FIRSTLAW_rungs_v1.md
nice -n 10 $PY w6_fig_v1.py --readout-dir $O --out $O/W6_FIRSTLAW_rungs_v1_fig
cd $O && md5sum * > MANIFEST_w6_v1.md5
echo done
