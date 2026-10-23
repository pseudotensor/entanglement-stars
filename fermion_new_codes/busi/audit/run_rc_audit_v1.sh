#!/bin/bash
# rc_audit_v1 on the box CPU (nice 10, at most 8 threads); reads stored json only
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
B=${QI2_ROOT}/solve_a_bulk
cd ${QI2_ROOT}/rc_v1
nice -n 10 python3 -I code/rc_audit_v1.py --tools ${QI2_ROOT}/tools --out-dir ${QI2_ROOT}/rc_v1 \
 --member M3c_s4_nk=$B/ladders/p8/runs/lad_M3c_s4_nk_m0p1_rs96_n3072_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36j/collapse_lad_M3c_s4_nk_m0p1_rs96_n3072_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36j.json \
 --member P64_s2=$B/v36d_relaunch/records/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json \
 --member RS32_s2=$B/pod_sync/pod1_engine/C3lad32r0b/collapse_c32_rs32_A_R2_tau2_ell2_prep4_avg_nk40r0b.json \
 --member S4_128_s4_nk_v36m=$B/ladders/p8/runs/lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36m/collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36m.json \
 --member S4_128_s4_nk_v36q=$B/ladders/r3/runs/lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36q/collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36q.json \
 2>&1 | tee rc_audit_v1.log
