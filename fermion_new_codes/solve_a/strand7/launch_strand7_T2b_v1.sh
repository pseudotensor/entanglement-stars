#!/bin/bash
# Task 2 job T2b (SPEC_task2_static_row_diagnostics.md sec. 3): strand7_shell_v5.py --t2b 1 (the monitors M8 to M11; M9's markers every 10th
# output) on self E16, h = 1, dt = 0.01 r_s, R_max = 64 r_s, T_end = 60 r_s0 (the record self_E16_R4_A's settings).  One CPU thread, nice,
# gated on the lane's busy processes (at most 15).  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
mkdir -p T2b
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 30; done
date -u "+T2b start %H:%M:%S UTC (self E16, --t2b 1)" >> T2b/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand7_shell_v5.py --run self \
  --E0 16 --R0-over-rs 4 --h 1 --Rmax-over-rs 64 --dt-over-rs 0.01 --T-over-rs 60 --t2b 1 --marker-every 10 --out T2b --tag self_E16_t2b > T2b/run.log 2>&1
echo "T2b exit $? $(date -u +%H:%M:%S)" >> T2b/status.txt
echo DONE >> T2b/status.txt
