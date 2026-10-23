#!/bin/bash
# C-capture v7 test 4 (builder analyst, 2026-10-16): the last maximum's background (slice 283, T/rs 35.375) through v7's re-solve and
# trapping radius before its production line runs (version 6 never ran this slice: its maximum line was withdrawn): both backgrounds, the
# absorbing port, omega/m 1.1, |kappa| 1, one reading step; --workers 2; the runner environment (BLAS 8 threads). nice 10.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CAP7TEST
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
nice -n 10 python code/engine_capture_v7.py --fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz \
  --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied --slice 283 --control --absorber \
  --omega-over-m 1.1 --kappa-cap 1 --T-max-over-rs 2 --workers 2 \
  --out-json CAP7TEST/test4_slice283.json --md CAP7TEST/test4_slice283.md > CAP7TEST/test4_slice283.log 2>&1
echo "exit $?" >> CAP7TEST/test4_slice283.log
