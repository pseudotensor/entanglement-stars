#!/bin/bash
# run_rd2_field_gate_v1.sh (busi, analyst; PERF RD-2/RD-5 self-test (a) field level and (b)): rd2_field_gate_v1.py on the prototype's two
# stored record sets, box CPU, nice 10, 8 threads, the QI2_FIELDS_* variables unset. Outputs RD2_build/gate/.
set -u
B=${QI2_ROOT}/RD2_build; PY=${QI2_ROOT}/python; CODE=${QI2_ROOT}/code16c
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM H2_PW_ONE H2_LOCAL_SPLINE PYTHONPATH
cd $B
nice -n 10 $PY -I code/rd2_field_gate_v1.py gate $CODE kit/engine_fields_v16d.py \
  ${QI2_ROOT}/collapse_lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq_fields.npz 30 --tag iso128 > logs/gate_iso128.log 2>&1
echo "iso128 rc $?" >> logs/gate_done.txt
nice -n 10 $PY -I code/rd2_field_gate_v1.py gate $CODE kit/engine_fields_v16d.py \
  ${QI2_ROOT}/collapse_RUNG2c40b_srf_fields_T0012.00.npz 20 --tag rung2 > logs/gate_rung2.log 2>&1
echo "rung2 rc $?" >> logs/gate_done.txt
