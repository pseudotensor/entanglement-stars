#!/bin/bash
# engine_shell0_convention_check_v2 (engine spec 38.3 (3)) on the production's 281 certified records: three parts as separate processes
# (4 threads each, nice 10; no fork), then the compose of their JSONs; the control (the stored stress evaluated) on every 10th record.
# Outputs SHELL0/shell0_convention_check_v2.md and .json (parts SHELL0/shell0_v2_part<i>.json), logs SHELL0/log_shell0_v2_*.log, status
# SHELL0/status.txt. Version 1's launcher ran a forked pool whose workers deadlocked (stopped 00:04:42 UTC). Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
A="--json H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --data COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz --cert CERT/certificate_v3_c19_rs64.json --control-every 10 --md SHELL0/shell0_convention_check_v2.md"
echo "shell0_convention_check_v2 start $(date -u +%H:%M:%S) UTC" >> SHELL0/status.txt
for i in 0 1 2; do
  nice -n 10 $PY code/engine_shell0_convention_check_v2.py $A --part $i --parts 3 --out-json SHELL0/shell0_v2_part$i.json > SHELL0/log_shell0_v2_part$i.log 2>&1 &
done
wait
echo "shell0_convention_check_v2 parts done $(date -u +%H:%M:%S) UTC" >> SHELL0/status.txt
nice -n 10 $PY code/engine_shell0_convention_check_v2.py $A --compose SHELL0/shell0_v2_part0.json,SHELL0/shell0_v2_part1.json,SHELL0/shell0_v2_part2.json --out-json SHELL0/shell0_convention_check_v2.json > SHELL0/log_shell0_v2_compose.log 2>&1
echo "shell0_convention_check_v2 exit $? $(date -u +%H:%M:%S) UTC" >> SHELL0/status.txt
