#!/bin/bash
# J2 version 4 (observables 13.3; obs_bw_modular_v4.py) at its production sizes on the CPU, lowest priority: W in {64, 128, 256}, L/W in {16, 32}
# (L to 8192), the profiles (0, 0.6) and (-0.5, 0.3) with both cuts and the sign-reversed profile, the flat and uniform-flow controls; 16 threads,
# nice 10. Usage (on the box, detached, from obs): nohup bash code/launch_obs_J2v4_v1.sh > J2v4/launcher_v1.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/obs || exit 1
mkdir -p J2v4
echo "J2v4 start $(date -u +%H:%M:%S) UTC (CPU, 16 threads)" >> status.txt
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 nice -n 10 ${QI2_ROOT}/python code/obs_bw_modular_v4.py --W 64 128 256 --LW 16 32 \
  --out-json J2v4/J2v4_main.json --md J2v4/J2v4_main.md > J2v4/J2v4_main.log 2>&1 < /dev/null
echo "J2v4 exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
