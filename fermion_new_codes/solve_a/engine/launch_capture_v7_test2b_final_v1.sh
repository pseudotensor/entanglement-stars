#!/bin/bash
# C-capture v7 test 2b (builder analyst, 2026-10-16): the final engine_capture_v7.py (its docstring completed after tests 1 to 4) on test 2's
# serial arguments; compared with test 2's serial JSON (the same numbers to the bit are expected). nice 10, BLAS 1 thread as test 2.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
md5sum code/engine_capture_v7.py > CAP7TEST/test2b_final_md5.txt
nice -n 10 python code/engine_capture_v7.py --fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied --slice 302 --control --absorber --omega-over-m 1.1 2 --kappa-cap 4 --T-max-over-rs 12 --out-json CAP7TEST/test2b_final_w1.json --md CAP7TEST/test2b_final_w1.md > CAP7TEST/test2b_final_w1.log 2>&1
echo "exit $?" >> CAP7TEST/test2b_final_w1.log
nice -n 10 python code/compare_capture_v7_runs_v1.py CAP7TEST/test2_logic_w1.json CAP7TEST/test2b_final_w1.json > CAP7TEST/test2b_compare.txt 2>&1
