#!/bin/bash
# J-PL greybody lead ladder (obs_plateau_v2.py, profile pgs with the clean continuation R_1 = r_s/16, w = 4): kappa a = 1/128 and 1/256,
# R_lead in {128, 256, 512, 1024, 2048} r_s, (mu, r) = (0.2, 0.25) and (0.4, 0.6); the same with the smooth outer saturation w_L = 0.05 R_lead
# (the lead-junction control; version 2's pgs_outer run used w_L = 0.25 R_lead, which pushed rho below zero inside: its exit 1).
# CPU, 4 threads, nice 10. Usage (on the box, detached): nohup bash code/launch_obs_JPL_v3.sh > JPL/launcher_v3.log 2>&1 < /dev/null &
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
run gb_ladder --profile pgs --kappa-a 0.0078125 0.00390625 --r 0.25 --mu 0.2 --extra 0.6:0.4 --R-lead 128 256 512 1024 2048 --cont 0.0625:4
run gb_ladder_outer --profile pgs --kappa-a 0.0078125 0.00390625 --r 0.25 --mu 0.2 --extra 0.6:0.4 --R-lead 128 256 512 1024 2048 --cont 0.0625:4 --outer-w 0.05
