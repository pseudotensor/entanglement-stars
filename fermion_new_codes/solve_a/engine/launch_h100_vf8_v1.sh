#!/bin/bash
# J-VF's decisive rerun (spec 18.2) on the H100 pod: engine_vacflow_v8 at r_s 32, t_r 32 r_s, hold 2 r_s, kappa 1..16, the full chain
# current, R2 along the ramp, readouts every 0.5 r_s (the benchmark: 3 readouts of kappa 4 in 3.7 s beside the primary).
# Usage (on the pod, detached): cd ${QI2_ROOT}/engine; nohup bash code/launch_h100_vf8_v1.sh > logs_h100_vf8_launch.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p VF8
echo "vf8 start $(date -u +%H:%M:%S) UTC"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /usr/bin/python3.11 code/engine_vacflow_v8.py --rs 32 --profile exact --m 0 --kmax 16 --tr 32 --hold 2 \
  --every 0.5 --chain-current full --gpu 0 --out VF8 --tag vf8_rs32_exact_m0_full_R2 > logs_h100_vf8_rs32_exact_m0_full_R2.log 2>&1 < /dev/null
echo "vf8 exit $? $(date -u +%H:%M:%S) UTC"
