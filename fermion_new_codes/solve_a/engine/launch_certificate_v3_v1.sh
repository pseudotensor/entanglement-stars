#!/bin/bash
# engine_certificate_v3 (engine spec 31.5: the icloud lane's two normalization corrections) on the delivered record data: first its
# homogeneous validation (N = 1: equal to version 2's); chain 1 the r_s 64 production (COLLAPSE_records_v1.1; its certified windows of 30.5
# re-read under the corrected normalization); chain 2 the r_s 32 members tau 16, 8, 4 and reading B at 4 r_s; the ladder's members follow
# when their record data exist. Box CPU, nice 10, 4 threads per chain. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
S=CERT/status_v3.txt
echo "CERT3 validate start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_certificate_v3.py --validate --md CERT/certificate_v3_validate.md > CERT/log_certificate_v3_validate.log 2>&1
echo "CERT3 validate exit $? $(date -u +%H:%M:%S) UTC" >> $S
run() {  # tag data
  echo "CERT3 $1 start $(date -u +%H:%M:%S) UTC" >> $S
  nice -n 10 $PY code/engine_certificate_v3.py --data $2 --md CERT/certificate_v3_$1.md --out-json CERT/certificate_v3_$1.json > CERT/log_certificate_v3_$1.log 2>&1
  echo "CERT3 $1 exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
( run c19_rs64 COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz ) &
( for t in tau8 tau16 tau4; do run set17_rs32_A_$t COLL/COLLAPSE_records_v1_set17_rs32_A_R2_${t}_ell4_avg_data.npz; done
  [ -f COLL/COLLAPSE_records_v1_set17_rs32_B_R2_tau8_ell4_R0x4_avg_data.npz ] && run set17_rs32_B_tau8 COLL/COLLAPSE_records_v1_set17_rs32_B_R2_tau8_ell4_R0x4_avg_data.npz ) &
wait
echo "CERT3 all DONE $(date -u +%H:%M:%S) UTC" >> $S
