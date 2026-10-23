#!/bin/bash
# run_posbound_box_v1.sh (solve builder 12; 54.2 (2), (3)): the graft positive bound on the box CPU for the c15 and c30 holds (16 threads each, nice 10)
B=${QI2_ROOT}/solve_a_bulk; C=$B/graft/code; O=$B/graft/posbound; PY=${QI2_ROOT}/python
REC=$B/horizon_v3/in/prod/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json; FLD=$B/horizon_v3/in/prod/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz; HS=$B/pod_sync/pod2_page/holdstates
cd $O
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 nice -n 10 $PY $C/page_graft_posbound_v1.py --record $REC --fields $FLD --init-state $HS/v6bA_s4_state_at_015.000.pt --freeze-at-rs 15 --LH 1.644586285442289e-06 --trunc 1 --threads 16 --out $O --tag posbound_c15 > logs/posbound_c15.log 2>&1 &
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 nice -n 10 $PY $C/page_graft_posbound_v1.py --record $REC --fields $FLD --init-state $HS/v6bA_s4_state_at_030.000.pt --freeze-at-rs 30 --LH 3.49132789793604e-06 --trunc 1 --threads 16 --out $O --tag posbound_c30 > logs/posbound_c30.log 2>&1 &
wait
