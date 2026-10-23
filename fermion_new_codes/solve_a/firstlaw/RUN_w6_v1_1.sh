#!/bin/bash
# RUN_w6_v1_1.sh: W6_FIRSTLAW_rungs_v1.1 on the box (CPU, nice 10): the K_1-weight reading of the identity, then version 1's producer extended,
# the notes, the changes list, the composition and the figure. Inputs unchanged from version 1 (the six v4b jsons, the v3b outputs, the records).
set -e
C=${QI2_ROOT}/qi2_w6_code; F=${QI2_ROOT}/firstlaw; O=$F/w6_rungs_v1_1
RD=$O/in/ReviewA_reading_W6_rungs_v1.md   # the box copy of deliveries/lead_gaps/ReviewA_reading_W6_rungs_v1.md (md5 ea17fbfdbc56fd5fb733c6e87051c332)
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p $O; cd $C
T2=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d; T4=c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40; T8=c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40
[ -s $O/w6_k1weight_v1_1.json ] || nice -n 10 $PY w6_k1weight_v1_1.py --out-dir $O --rung s8_engine=$F/sigma8_rung_v4b_ek/firstlaw_records_v4b_$T8.json \
  --rung s4_engine=$F/sigma4_rung_v4b_ek/firstlaw_records_v4b_$T4.json --rung s2_engine=$F/sigma2_point_v4b_ek/firstlaw_records_v4b_$T2.json
nice -n 10 $PY w6_readout_v1_1.py --ladders ${QI2_ROOT}/ladders_readout_v1_1.json --k1 $O/w6_k1weight_v1_1.json --phases 14.75,17.5 --out-dir $O \
  --rung s2_engine=2=$F/sigma2_point_v4b_ek/firstlaw_records_v4b_$T2.json --rung s4_engine=4=$F/sigma4_rung_v4b_ek/firstlaw_records_v4b_$T4.json \
  --rung s8_engine=8=$F/sigma8_rung_v4b_ek/firstlaw_records_v4b_$T8.json \
  --rung s2_analytic=2=$F/sigma2_point_v4b/firstlaw_records_v4b_$T2.json --rung s4_analytic=4=$F/sigma4_rung_v4b/firstlaw_records_v4b_$T4.json \
  --rung s8_analytic=8=$F/sigma8_rung_v4b/firstlaw_records_v4b_$T8.json > $O/readout_v1_1.log 2>&1
nice -n 10 $PY w6_notes_v1_1.py --readout $O/w6_readout_v1_1.json --code-dir $C --reading $RD --out $O/w6_notes_v1_1.json
nice -n 10 $PY w6_changes_v1_1.py --readout $O/w6_readout_v1_1.json --reading $RD --out $O/w6_changes_v1_1.json
nice -n 10 $PY w6_compose_v1_1.py --readout $O/w6_readout_v1_1.json --notes $O/w6_notes_v1_1.json --changes $O/w6_changes_v1_1.json --out $O/W6_FIRSTLAW_rungs_v1.1.md
nice -n 10 $PY w6_fig_v1_1.py --readout-dir $O --out $O/W6_FIRSTLAW_rungs_v1.1_fig
cd $O && md5sum w6_k1weight_v1_1.json w6_readout_v1_1.json w6_readout_v1_1.npz w6_notes_v1_1.json w6_changes_v1_1.json W6_FIRSTLAW_rungs_v1.1.md W6_FIRSTLAW_rungs_v1.1_fig.png W6_FIRSTLAW_rungs_v1.1_fig.pdf > MANIFEST_w6_v1_1.md5
echo done
