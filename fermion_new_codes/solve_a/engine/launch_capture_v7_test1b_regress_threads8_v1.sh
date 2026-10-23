#!/bin/bash
# C-capture v7 test 1b (builder analyst, 2026-10-16): test 1 with the lane runner's BLAS setting (8 threads, as version 6's run had):
# does the re-solve's E then equal version 6's, and v7's I and monitor at t = 1..5 r_s with it? nice 10; outputs in CAP7TEST/.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CAP7TEST
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
nice -n 10 python code/engine_capture_v7.py --fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz \
  --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied --slice -1 --absorber \
  --omega-over-m 1.1 --kappa-cap 3 --T-max-over-rs 6 \
  --out-json CAP7TEST/test1b_regress_last_t8.json --md CAP7TEST/test1b_regress_last_t8.md > CAP7TEST/test1b_regress_last_t8.log 2>&1
echo "exit $?" >> CAP7TEST/test1b_regress_last_t8.log
nice -n 10 python code/compare_capture_v7_v6_monitor_v1.py CAP/capture_v6_prod19_last.json CAP7TEST/test1b_regress_last_t8.json > CAP7TEST/test1b_compare.txt 2>&1
python -c "
import json
for p in ('CAP/capture_v6_prod19_last.json', 'CAP7TEST/test1_regress_last.json', 'CAP7TEST/test1b_regress_last_t8.json'):
    d = json.load(open(p)); print(p, 'E %.17g R_t %.17g r_s %.17g' % (d['E'], d['R_t'], d['r_s']))
" >> CAP7TEST/test1b_compare.txt 2>&1
