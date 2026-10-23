#!/bin/bash
# C-capture v7 test 2 (builder analyst, 2026-10-16): the logic end to end on the last minimum (slice 302): both backgrounds, the
# absorbing port, omega/m 1.1 and 2, |kappa| to 4, reading times 2 to 12 r_s; serial, then --workers 2 on the same arguments; compared.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CAP7TEST
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARGS="--fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied --slice 302 --control --absorber --omega-over-m 1.1 2 --kappa-cap 4 --T-max-over-rs 12"
nice -n 10 python code/engine_capture_v7.py $ARGS --out-json CAP7TEST/test2_logic_w1.json --md CAP7TEST/test2_logic_w1.md > CAP7TEST/test2_logic_w1.log 2>&1
echo "exit $?" >> CAP7TEST/test2_logic_w1.log
nice -n 10 python code/engine_capture_v7.py $ARGS --workers 2 --out-json CAP7TEST/test2_logic_w2.json --md CAP7TEST/test2_logic_w2.md > CAP7TEST/test2_logic_w2.log 2>&1
echo "exit $?" >> CAP7TEST/test2_logic_w2.log
nice -n 10 python code/compare_capture_v7_runs_v1.py CAP7TEST/test2_logic_w1.json CAP7TEST/test2_logic_w2.json > CAP7TEST/test2_compare.txt 2>&1
