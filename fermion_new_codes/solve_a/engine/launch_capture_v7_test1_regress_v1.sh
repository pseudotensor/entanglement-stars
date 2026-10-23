#!/bin/bash
# C-capture v7 test 1 (builder analyst, 2026-10-16): the regression against version 6's JSON on the last record (slice 320):
# omega/m 1.1, |kappa| 1 to 3, solved slice only, reading times 2, 4, 6 r_s, the absorbing port on; v7's I and monitor at t = 1..5 r_s
# must equal version 6's (both use 1 r_s Chebyshev steps there). nice 10, BLAS 1 thread; outputs in CAP7TEST/.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CAP7TEST
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 python code/engine_capture_v7.py --fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz \
  --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied --slice -1 --absorber \
  --omega-over-m 1.1 --kappa-cap 3 --T-max-over-rs 6 \
  --out-json CAP7TEST/test1_regress_last.json --md CAP7TEST/test1_regress_last.md > CAP7TEST/test1_regress_last.log 2>&1
echo "exit $?" >> CAP7TEST/test1_regress_last.log
nice -n 10 python code/compare_capture_v7_v6_monitor_v1.py CAP/capture_v6_prod19_last.json CAP7TEST/test1_regress_last.json > CAP7TEST/test1_compare.txt 2>&1
