#!/bin/bash
# engine_rows_floor_v1 --mode records (engine spec 27.2 and 27.7: "engine_rows_floor_v1 on the tau 4 and ell 8 controls' records at T/rs 5 and
# 10 (new output files)"): the r_s 64 tau 4 control (c19_rs64_A_R2_tau4_ell4_avg; fields v14; the centred difference over 8 time units) from a
# validated snapshot taken on the pod at 03:35 UTC (CCT/snap_c19_tau4_v1, records to T/rs 21.875; the run continues to 40), and the r_s 64
# ell 8 control (c24_rs64_A_R2_tau8_ell8_avg; fields v16; ended at T/rs 40) from its synced copy; each with its native handoff. Box CPU, one at
# a time, nice 10, 8 threads. Builder analyst.
cd ${QI2_ROOT}/engine
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=${QI2_ROOT}/python
nice -n 10 $PY code/engine_rows_floor_v1.py --mode records --run CCT/snap_c19_tau4_v1/c19_rs64_A_R2_tau4_ell4_avg --handoff CCT/snap_c19_tau4_v1/collapse_c19_rs64_A_R2_tau4_ell4_avg_handoff.npz --times 5,10 --md CCT/rows_floor_c19_tau4_5_10_v1.md --out-json CCT/rows_floor_c19_tau4_5_10_v1.json > CCT/log_rows_floor_c19_tau4.log 2>&1; echo "CCT rows floor tau 4 control 5 10 exit $? $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
nice -n 10 $PY code/engine_rows_floor_v1.py --mode records --run H100b/C3prod24/c24_rs64_A_R2_tau8_ell8_avg --handoff H100b/C3prod24/collapse_c24_rs64_A_R2_tau8_ell8_avg_handoff.npz --times 5,10 --md CCT/rows_floor_c24_ell8_5_10_v1.md --out-json CCT/rows_floor_c24_ell8_5_10_v1.json > CCT/log_rows_floor_c24_ell8.log 2>&1; echo "CCT rows floor ell 8 control 5 10 exit $? $(date -u +%H:%M:%S) UTC" >> CCT/status.txt
