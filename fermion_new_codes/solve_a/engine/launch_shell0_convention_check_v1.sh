#!/bin/bash
# engine_shell0_convention_check_v1 (engine spec 38.3 (3)) on the production's 281 certified records; box CPU, nice 10, 3 workers of 4 threads
# (12 threads beside engine_core_slices_v3's 4: this lane's 16); the control (the stored stress evaluated) on every 10th record. Outputs
# SHELL0/shell0_convention_check_v1.md and .json, log SHELL0/log_shell0_v1.log, status SHELL0/status.txt. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "shell0_convention_check_v1 start $(date -u +%H:%M:%S) UTC" >> SHELL0/status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_shell0_convention_check_v1.py --json H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --data COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz --cert CERT/certificate_v3_c19_rs64.json --workers 3 --control-every 10 --md SHELL0/shell0_convention_check_v1.md --out-json SHELL0/shell0_convention_check_v1.json > SHELL0/log_shell0_v1.log 2>&1
echo "shell0_convention_check_v1 exit $? $(date -u +%H:%M:%S) UTC" >> SHELL0/status.txt
