#!/bin/bash
# launch_replay_v9_cputime_p2_v2.sh (builder analyst): a timing and exactness test only, not a deliverable. Version 1 timed engine_replay_v9
# on pod 2's CPU without replay v2's stored masks (no --compare-prefix), so its dust/pairs split could not be compared; this version runs the
# set17 tau 8 member exactly as the S5 launcher does (--compare-prefix REPLAY2/replay2_set17_rs32_A_R2_tau8_ell4_avg), its first 12 records,
# on the CPU (16 threads, nice 10), to see whether the CPU replay stays within the 1e-8 acceptance against the stored GPU records.
# Outputs in CERT3v9cpu_t2/. Started directly by nohup.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CERT3v9cpu_t2
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
echo "start $(date -u +%H:%M:%S) UTC" > CERT3v9cpu_t2/status.txt
nice -n 10 python code/engine_replay_v9.py --run C3set/set17_rs32_A_R2_tau8_ell4_avg --records 12 --compare-prefix REPLAY2/replay2_set17_rs32_A_R2_tau8_ell4_avg --md CERT3v9cpu_t2/t.md --out-json CERT3v9cpu_t2/t.json --out-npz CERT3v9cpu_t2/t > CERT3v9cpu_t2/log.log 2>&1
echo "exit $? $(date -u +%H:%M:%S) UTC" >> CERT3v9cpu_t2/status.txt
