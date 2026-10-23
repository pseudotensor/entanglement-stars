#!/bin/bash
# C-capture v7 test 3p (builder analyst, 2026-10-16): test 3 with --workers 8 (8 forked single-threaded processes); the same arguments
# otherwise (the runner environment, BLAS 8 threads), so the step times compare and the batch rule (speculative channels beyond the stop dropped) is exercised at omega/m 1.1. nice 10.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CAP7TEST
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
nice -n 10 python code/engine_capture_v7.py --fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz \
  --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied --slice 302 --absorber \
  --omega-over-m 2 1.1 --T-max-over-rs 4 --workers 8 \
  --out-json CAP7TEST/test3_steptime_w8.json --md CAP7TEST/test3_steptime_w8.md > CAP7TEST/test3_steptime_w8.log 2>&1
echo "exit $?" >> CAP7TEST/test3_steptime_w8.log
