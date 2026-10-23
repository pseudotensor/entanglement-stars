#!/bin/bash
# run_rd2b_field_gate_v1.sh (busi, analyst; PERF RD-2b self-test (a) field level and (b)): rd2b_field_gate_v1.py on the stored fallback
# records and on ISO_128's last 30 (intact) records, box CPU, nice 10, 8 threads, the QI2_FIELDS_* variables unset. Outputs RD2b_build/gate/.
set -u
B=${QI2_ROOT}/RD2b_build; PY=${QI2_ROOT}/python; CODE=${QI2_ROOT}/code16c
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE PYTHONPATH
cd $B
ISO=${QI2_ROOT}/collapse_lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq_fields.npz
G="$PY -I -W ignore code/rd2b_field_gate_v1.py gate $CODE kit/engine_fields_v16d.py kit/engine_fields_v16e.py"
nice -n 10 $G ../RD2_build/h100f/v2/S2/I/collapse_RUNG2c40_v36q_rd2I_fields.npz --tag s2replay --recs 17,18,19,20,21,22,23,24 > logs/gate_s2replay.log 2>&1; echo "s2replay rc $?" >> logs/gate_done.txt
nice -n 10 $G ${QI2_ROOT}/collapse_RUNG2c40_v36q_fields.npz --tag rung2prod_neg --negcentre > logs/gate_rung2prod_neg.log 2>&1; echo "rung2prod_neg rc $?" >> logs/gate_done.txt
nice -n 10 $G $ISO --tag iso128_neg --negcentre > logs/gate_iso128_neg.log 2>&1; echo "iso128_neg rc $?" >> logs/gate_done.txt
nice -n 10 $G $ISO --tag iso128_last30 --last 30 > logs/gate_iso128_last30.log 2>&1; echo "iso128_last30 rc $?" >> logs/gate_done.txt
