#!/bin/bash
# J-PL version 2 (obs_plateau_v2.py; SPEC_final4b_observables.md sec. 9.1) on the CPU, 4 threads, nice 10: the exact exterior with the
# analytic interior continuation at kappa a = 1/128, 1/256, 1/512 (the continuation ladder R_1/r_s:w in {1/8:4, 1/16:4, 1/8:3}, the lead
# ladder R_lead in {32, 64, 128, 256} r_s; the greybody points inside the grid); the collapse's sizes kappa a = 1/32 and 1/64 with the best
# available continuations (their spread is their error); the smooth outer saturation as the lead-junction control.
# Usage (on the box, detached): nohup bash code/launch_obs_JPL_v2.sh > JPL/launcher_v2.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/obs
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
run() {
  tag=$1; shift
  echo "J-PL v2 $tag start $(date -u +%H:%M:%S) UTC (CPU, 4 threads)" >> status.txt
  nice -n 10 $PY code/obs_plateau_v2.py "$@" --out-json JPL/jpl2_$tag.json --md JPL/jpl2_$tag.md > JPL/jpl2_$tag.log 2>&1 < /dev/null
  echo "J-PL v2 $tag exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
}
run pgs_main --profile pgs --kappa-a 0.0078125 0.00390625 0.001953125 --r 0.25 0.5 1 --mu 0 0.2 0.4 --extra 0.6:0.4 --R-lead 32 64 128 256 --cont 0.125:4 0.0625:4 0.125:3
run pgs_rs32 --profile pgs --kappa-a 0.015625 --r 0.25 0.5 1 --mu 0 0.2 0.4 --extra 0.6:0.4 --R-lead 64 256 --cont 0.125:2.5 0.0625:3 0.25:2
run pgs_rs16 --profile pgs --kappa-a 0.03125 --r 0.25 0.5 1 --mu 0 0.2 0.4 --extra 0.6:0.4 --R-lead 64 256 --cont 0.125:1.5 0.0625:2 0.25:1
run pgs_outer --profile pgs --kappa-a 0.0078125 0.00390625 --r 0.25 --mu 0.2 --extra 0.6:0.4 --R-lead 32 64 128 256 --cont 0.125:4 --outer-w 0.25
