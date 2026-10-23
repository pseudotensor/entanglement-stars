#!/bin/bash
# A smoke test of engine_capture_v9 on pod 1 (slice 302, three channels, reading times to 8 r_s, 2 workers; outputs CAP9TEST/), run in the
# foreground of its own call; it does not touch the lanes' queues. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CAP9TEST
F="--fields C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --records C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields-key applied"
OMP_NUM_THREADS=8 nice -n 10 timeout 1500 python code/engine_capture_v9.py $F --slice 302 --control --absorber --workers 2 --kappa-cap 3 --T-max-over-rs 8 --out-json CAP9TEST/smoke_v9.json --md CAP9TEST/smoke_v9.md > CAP9TEST/smoke_v9.log 2>&1
echo "exit $?"
