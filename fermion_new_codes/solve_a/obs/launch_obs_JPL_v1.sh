#!/bin/bash
# J-PL (SPEC_final4b_observables.md sec. 9.1) on the CPU, 4 threads, nice 10: the tanh control and its coefficients; the exact profile with
# the lead ladder R_lead in {32, 64, 128, 256} r_s and the inner-lead ladder R_in in {1/2, 1/4} r_s at kappa a = 1/32, 1/64, 1/128; the
# greybody code checks at kappa a = 1/128 and 1/256 ((mu, r) = (0.2, 0.25) and (0.4, 0.6)).
# Usage (on the box, detached): nohup bash code/launch_obs_JPL_v1.sh > JPL/launcher_v1.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/obs
mkdir -p JPL
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "J-PL tanh start $(date -u +%H:%M:%S) UTC (CPU, 4 threads)" >> status.txt
nice -n 10 $PY code/obs_plateau_v1.py --profile tanh --kappa-a 0.125 0.0625 0.03125 0.015625 --r 0.25 0.5 1 --mu 0 0.2 0.4 --v-out-min 1e-6 1e-8 --out-json JPL/jpl_tanh_v1.json --md JPL/jpl_tanh_v1.md > JPL/jpl_tanh_v1.log 2>&1 < /dev/null
echo "J-PL tanh exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
echo "J-PL pg start $(date -u +%H:%M:%S) UTC (CPU, 4 threads)" >> status.txt
nice -n 10 $PY code/obs_plateau_v1.py --profile pg --kappa-a 0.03125 0.015625 0.0078125 --r 0.25 0.5 1 --mu 0 0.2 0.4 --extra 0.6:0.4 --R-lead 32 64 128 256 --R-in 0.5 0.25 --W-in 0.25 --out-json JPL/jpl_pg_v1.json --md JPL/jpl_pg_v1.md > JPL/jpl_pg_v1.log 2>&1 < /dev/null
echo "J-PL pg exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
echo "J-PL greybody start $(date -u +%H:%M:%S) UTC (CPU, 4 threads)" >> status.txt
nice -n 10 $PY code/obs_plateau_v1.py --profile pg --kappa-a 0.0078125 0.00390625 --r 0.25 --mu 0.2 --extra 0.6:0.4 --R-lead 32 64 128 256 --R-in 0.5 0.25 --W-in 0.25 --out-json JPL/jpl_greybody_v1.json --md JPL/jpl_greybody_v1.md > JPL/jpl_greybody_v1.log 2>&1 < /dev/null
echo "J-PL greybody exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
