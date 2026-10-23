#!/bin/bash
# N4 then N5 on the box CPU (one at a time, at most 8 threads, nice 10). Outputs in ${QI2_ROOT}/.
set -u
PY=${QI2_ROOT}/python
C=${QI2_ROOT}/n345
cd ${QI2_ROOT}/n345
nice -n 10 $PY $C/n4_b_zero_v1.py --tag v1 > n4_b_zero_v1.log 2>&1
L=${QI2_ROOT}/r1
TAG=lad_ISO_64_s4_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n
nice -n 10 $PY $C/n5_surface_identity_v1.py --slice $L/slices/orbitals_${TAG}_T0000.00.npz \
  --fields $L/runs/$TAG/collapse_${TAG}_fields.npz --json $L/runs/$TAG/collapse_${TAG}.json --tag ISO64s4_T0000 > n5_ISO64s4_T0000.log 2>&1
nice -n 10 $PY $C/n5_surface_identity_v1.py --slice $L/slices/orbitals_${TAG}_T0011.00.npz \
  --fields $L/runs/$TAG/collapse_${TAG}_fields.npz --json $L/runs/$TAG/collapse_${TAG}.json --tag ISO64s4_T0011 > n5_ISO64s4_T0011.log 2>&1
echo DONE > n45_runs_v1.done
