#!/bin/bash
# Gamma_j(omega) for J3 version 3 (SPEC_final4b_observables.md sec. 8.1) by stationary scattering (obs_greybody_radial_v1.py), CPU, 4 threads,
# nice 10: r_s in {256, 512}; j in {1/2, 3/2}; m/kappa in {0, 1/4}; the sign -1 at j = 1/2, m = 0; R_lead in {32, 64} r_s; the continuation
# ladder R_1/r_s:w in {1/16:4, 1/8:4}; 64 frequencies per decade over [kappa/4, 8 kappa].
# Usage (on the box, detached): nohup bash code/launch_obs_GB_v1.sh > GB/launcher_v1.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/obs
mkdir -p GB
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "GB radial main start $(date -u +%H:%M:%S) UTC (CPU, 4 threads)" >> status.txt
nice -n 10 $PY code/obs_greybody_radial_v1.py --rs 256 512 --j 0.5 1.5 --m-over-kappa 0 0.25 --signs +1 --R-lead 32 64 --cont 0.0625:4 0.125:4 --out-json GB/gb_main_v1.json --md GB/gb_main_v1.md > GB/gb_main_v1.log 2>&1 < /dev/null
echo "GB radial main exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
echo "GB radial sign start $(date -u +%H:%M:%S) UTC (CPU, 4 threads)" >> status.txt
nice -n 10 $PY code/obs_greybody_radial_v1.py --rs 256 512 --j 0.5 --m-over-kappa 0 --signs -1 --R-lead 32 64 --cont 0.0625:4 --out-json GB/gb_sign_v1.json --md GB/gb_sign_v1.md > GB/gb_sign_v1.log 2>&1 < /dev/null
echo "GB radial sign exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
