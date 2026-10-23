#!/bin/bash
# run_w6_v4c.sh: W6's producer v4c (the K_1 momentum weight; ReviewA_reading_W6_rungs_v1.md R1) on the three (0.2, 64) members, both kernel routes,
# then the comparison against each member's v4b json. Box CPU, nice 10, 8 threads; outputs ${QI2_ROOT}/v4c<rung>_<route>/; status in run_w6_v4c.status.
C=${QI2_ROOT}/w6w7; F=${QI2_ROOT}/firstlaw; O=${QI2_ROOT}/v4c; E=${QI2_ROOT}/code
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p $O; cd $C
T2=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d; P2=${QI2_ROOT}/collapse_$T2
T4=c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40; P4=${QI2_ROOT}/collapse_$T4
T8=c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40; P8=${QI2_ROOT}/collapse_$T8
run() {  # rung-label v3b-dir tag records-base v4b-dir-stem
  for K in engine analytic; do
    if [ $K = engine ]; then B=$F/${5}_v4b_ek; else B=$F/${5}_v4b; fi
    D=$O/${1}_$K
    timeout 3600 nice -n 10 $PY firstlaw_records_v4c.py --v3b-json $F/$2/firstlaw_records_v3b_$3.json --v3b-cache $F/$2/firstlaw_dense_v3b_${3}_cache.npz \
      --json $4.json --npz ${4}_fields.npz --handoff ${4}_handoff.npz --out $D --threads 8 --kernel $K --v3b-dir $E --engine-path $E > $D.log 2>&1
    echo "$1 $K exit $? $(date -u +%FT%TZ)" >> $O/run_w6_v4c.status
    [ -s $B/firstlaw_records_v4b_$3.json ] && nice -n 10 $PY w6_v4c_compare.py $B/firstlaw_records_v4b_$3.json $D/firstlaw_records_v4c_$3.json > $D.compare.log 2>&1
  done
}
run s2 sigma2_point_v3b $T2 $P2 sigma2_point
run s4 sigma4_rung_v3b $T4 $P4 sigma4_rung
run s8 sigma8_rung_v3b $T8 $P8 sigma8_rung
cd $O && md5sum */firstlaw_records_v4c_*.json > MANIFEST_w6_v4c.md5
echo "done $(date -u +%FT%TZ)" >> $O/run_w6_v4c.status
