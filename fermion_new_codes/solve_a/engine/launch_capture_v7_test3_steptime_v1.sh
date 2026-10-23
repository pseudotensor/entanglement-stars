#!/bin/bash
# C-capture v7 test 3 (builder analyst, 2026-10-16): one probe's step timed on pod 1, on the last minimum (slice 302), the solved slice,
# the absorbing port on, each probe's full channel list (version 6's rule, cap 200), two reading steps (2 and 4 r_s: the first builds and
# projects every packet, the second is a pure 2 r_s advance). Serial in the lane runner's environment (BLAS 8 threads). nice 10.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CAP7TEST
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
nice -n 10 python code/engine_capture_v7.py --fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz \
  --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied --slice 302 --absorber \
  --omega-over-m 2 1.1 --T-max-over-rs 4 \
  --out-json CAP7TEST/test3_steptime_w1.json --md CAP7TEST/test3_steptime_w1.md > CAP7TEST/test3_steptime_w1.log 2>&1
echo "exit $?" >> CAP7TEST/test3_steptime_w1.log
