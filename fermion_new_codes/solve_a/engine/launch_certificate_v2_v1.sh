#!/bin/bash
# engine_certificate_v2 (engine spec 28.2, S1's CPU part) on the delivered record data: chain 1 the r_s 64 production (COLLAPSE_records_v1.1,
# 321 records, fields v12 then v14 by segment); chain 2 the r_s 32 members tau 16, 8, 4 and reading B at 4 r_s (COLLAPSE_records_v1, v17,
# fields v12). Box CPU, nice 10, 4 threads per chain. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
S=CERT/status.txt
run() {  # tag data
  echo "CERT $1 start $(date -u +%H:%M:%S) UTC" >> $S
  nice -n 10 $PY code/engine_certificate_v2.py --data $2 --md CERT/certificate_v2_$1.md --out-json CERT/certificate_v2_$1.json > CERT/log_certificate_v2_$1.log 2>&1
  echo "CERT $1 exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
( run c19_rs64 COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz ) &
( for t in tau8 tau16 tau4; do run set17_rs32_A_$t COLL/COLLAPSE_records_v1_set17_rs32_A_R2_${t}_ell4_avg_data.npz; done
  [ -f COLL/COLLAPSE_records_v1_set17_rs32_B_R2_tau8_ell4_R0x4_avg_data.npz ] && run set17_rs32_B_tau8 COLL/COLLAPSE_records_v1_set17_rs32_B_R2_tau8_ell4_R0x4_avg_data.npz ) &
wait
echo "CERT all DONE $(date -u +%H:%M:%S) UTC" >> $S
