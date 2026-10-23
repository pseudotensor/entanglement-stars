#!/bin/bash
# After the r_s 32 tau 2 member ends (engine spec 25.2): the four-member tau table, the Gauss balance across tau with reading B, and the
# tau 2 member's dense max |v|/N summary (engine_records_summary_v5), on the pod, one after another, nice 10. Builder analyst.
cd ${QI2_ROOT}/engine
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
until grep -q "lane2c c24_rs32_tau2 exit" h100b_status.txt; do sleep 30; done
mkdir -p CHK SUM5
nice -n 10 python code/engine_tau_table_v2.py --runs C3set/set17_rs32_A_R2_tau16_ell4_avg C3set/set17_rs32_A_R2_tau8_ell4_avg C3set/set17_rs32_A_R2_tau4_ell4_avg C3set24/c24_rs32_A_R2_tau2_ell4_avg --times 16,18,20,24,30,35,40 --md CHK/tau_table_v2_final.md > CHK/log_tau_table_v2_final.log 2>&1
nice -n 10 python code/engine_gauss_tau_v1.py --members "tau16:C3set/set17_rs32_A_R2_tau16_ell4_avg:REPLAY2/replay2_set17_rs32_A_R2_tau16_ell4_avg_handoff.npz,tau8:C3set/set17_rs32_A_R2_tau8_ell4_avg:REPLAY2/replay2_set17_rs32_A_R2_tau8_ell4_avg_handoff.npz,tau4:C3set/set17_rs32_A_R2_tau4_ell4_avg:REPLAY2/replay2_set17_rs32_A_R2_tau4_ell4_avg_handoff.npz,tau2:C3set24/c24_rs32_A_R2_tau2_ell4_avg:C3set24/collapse_c24_rs32_A_R2_tau2_ell4_avg_handoff.npz,B_tau8:C3set/set17_rs32_B_R2_tau8_ell4_R0x4_avg:REPLAY2/replay2_set17_rs32_B_R2_tau8_ell4_R0x4_avg_handoff.npz" --windows 0-16,16-22,22-30,30-40,0-40 --md CHK/gauss_tau_v1.md > CHK/log_gauss_tau_v1.log 2>&1
echo "tau2 tables done $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
nice -n 10 python code/engine_records_summary_v5.py --run C3set24/c24_rs32_A_R2_tau2_ell4_avg --from 0 --md SUM5/summary_v5_c24_rs32_A_R2_tau2_ell4_avg.md > SUM5/log_tau2.log 2>&1
echo "tau2 summary v5 exit $? $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
