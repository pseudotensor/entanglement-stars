#!/bin/bash
# The goal of 19:57 UTC (the lead's review): "the exact-profile plateau at kappa a = 1/16, 1/32, 1/64 and 1/128 with the light massive probe
# and the leads matched to the sqrt(R) phase tail" on the idle CPU threads. obs_plateau_v2 (the J-PL driver) on the pgs profile (the exact
# Painleve-Gullstrand exterior, the analytic softplus interior continuation, the smooth outer saturation at 0.05 R_lead), four
# continuations per kappa a (their spread is the row's interior contamination: per the J-PL record a continuation is clean only for
# R_t - R_1 >= 14 w with w >= 4, i.e. kappa a <= 1/128), R_lead 256 and 1024 r_s, r = 0.25, 0.5, 1 (and 0.6), m/kappa = 0, 0.2, 0.4.
# CPU, 8 threads, nice 10. Usage (on the box, detached, from obs): nohup bash code/launch_obs_jpl5_exact_ladder_v1.sh > JPL/launcher_jpl5_v1.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/obs || exit 1
PY=${QI2_ROOT}/python
echo "JPL5 exact-profile ladder start $(date -u +%H:%M:%S) UTC (CPU, 8 threads)" >> status.txt
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10 $PY code/obs_plateau_v2.py --profile pgs --kappa-a 0.0625 0.03125 0.015625 0.0078125 \
  --cont 0.125:4 0.0625:4 0.125:3 0.25:2 --outer-w 0.05 --R-lead 256 1024 --r 0.25 0.5 1.0 --mu 0.0 0.2 0.4 --extra 0.6:0.4 \
  --out-json JPL/jpl5_exact_ladder_v1.json --md JPL/jpl5_exact_ladder_v1.md > JPL/jpl5_exact_ladder_v1.log 2>&1 < /dev/null
echo "JPL5 exact-profile ladder exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
