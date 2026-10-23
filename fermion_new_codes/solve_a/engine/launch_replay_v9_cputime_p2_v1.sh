#!/bin/bash
# launch_replay_v9_cputime_p2_v1.sh (builder analyst): a timing only, not a deliverable. engine_replay_v9 on the second H100 pod's host CPU,
# because S5's GPU run (replay9_s5_p2b on p2lane2) slowed about 20-fold after 20:17 UTC while the GPU is shared. The r_s 32 set17 tau 8
# member, its first 3 records, 16 threads, nice 10. Outputs in CERT3v9cpu_t/. Started directly by nohup, not by a queue runner.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p CERT3v9cpu_t
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
echo "start $(date -u +%H:%M:%S) UTC" > CERT3v9cpu_t/status.txt
nice -n 10 python code/engine_replay_v9.py --run C3set/set17_rs32_A_R2_tau8_ell4_avg --records 3 --md CERT3v9cpu_t/t.md --out-json CERT3v9cpu_t/t.json --out-npz CERT3v9cpu_t/t > CERT3v9cpu_t/log.log 2>&1
echo "exit $? $(date -u +%H:%M:%S) UTC" >> CERT3v9cpu_t/status.txt
