#!/bin/bash
# inbox #455 (ReviewA_found7_v1 sec. 3, ReviewB_found7_v1 sec. 3): after run_ls_v6.sh (third q) and run_ls_v7.sh (lambda = 3).
# Box CPU, nice 10, at most 8 threads in all (the 8-thread step alone; the post-processing steps one thread each, at most 4 at once).
# sigma/a = 4 inputs: the resume-mirror copy (json and fields md5 equal to v1.1's recorded d7d69adc... and 01444a39...);
# sigma/a = 8 inputs: pod_sync. Nothing under ${QI2_ROOT}/v36d_from0 is read.
set -u
PY=${QI2_ROOT}/python
D=${QI2_ROOT}/localsea
cd ${QI2_ROOT}/localsea
M4=${QI2_ROOT}/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40
P8=${QI2_ROOT}/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40
mkdir -p v12/s4a v12/s4b
while [ ! -e run_ls_v6.done ]; do sleep 60; done
nice -n 10 $PY -B $D/ls_rate3q_read_v1.py > logs/rate3q_read_v1.log 2>&1   # milliseconds of numpy
while [ ! -e run_ls_v7.done ]; do sleep 60; done
nice -n 10 $PY -B $D/ls_chain_kmax_v1.py --audit_npz audit/ls_chain_audit_v1_s8_T16.npz --K 40 --Rt 52.91344971597223 --out audit --tag s8_T16_K40 > logs/kmax_s8_T16_K40.log 2>&1
nice -n 10 $PY -B $D/ls_reviewbA_3d_v3.py --json4 $M4.json --fields4 ${M4}_fields.npz --json8 $P8.json --fields8 ${P8}_fields.npz --out v12 --tags s8_T14,s8_T16 > logs/reviewbA_3d_v3_s8.log 2>&1 &
nice -n 10 $PY -B $D/ls_reviewbA_3d_v3.py --json4 $M4.json --fields4 ${M4}_fields.npz --json8 $P8.json --fields8 ${P8}_fields.npz --out v12/s4a --tags s4_T07,s4_T13,s4_T14 > logs/reviewbA_3d_v3_s4a.log 2>&1 &
nice -n 10 $PY -B $D/ls_reviewbA_3d_v3.py --json4 $M4.json --fields4 ${M4}_fields.npz --json8 $P8.json --fields8 ${P8}_fields.npz --out v12/s4b --tags s4_T15,s4_T16 > logs/reviewbA_3d_v3_s4b.log 2>&1 &
nice -n 10 $PY -B $D/ls_audit_read_v2.py --parts rates --json4 $M4.json --fields4 ${M4}_fields.npz --handoff4 ${M4}_handoff.npz \
  --json8 $P8.json --fields8 ${P8}_fields.npz --handoff8 ${P8}_handoff.npz --out v12 > logs/audit_read_v2_rates_v12.log 2>&1 &
nice -n 10 $PY -B $D/ls_scale_read_v3.py --json4 $M4.json --fields4 ${M4}_fields.npz --out v12 > logs/scale_read_v3.log 2>&1 &
wait
echo DONE > run_ls_v8.done
