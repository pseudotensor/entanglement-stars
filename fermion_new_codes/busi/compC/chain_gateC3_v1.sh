#!/bin/bash
# gate C on engine_twofield_v3 (the condition of the ruling on gate D's (5): Part 1 bitwise on v3, whose changes are light-only), both members in
# sequence, each at a 1-minute load below 90 (the remote-box rule); gate_regression_v2b with --v36 engine_twofield_v3.py.
R=${QI2_ROOT}/compC; C=${QI2_ROOT}/compC; E=${QI2_ROOT}/compC_engine_v36
export BUSI_ENGINE_DIR=$E BUSI_PAGE_DIR=$E PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=${QI2_ROOT}/python
for M in s4 s2; do
  case $M in
    s4) A="--prod-dir ${QI2_ROOT}/C3lad32 --prod-tag c31_rs32_A_R2_tau2_ell4_prep4_avg"; X="--both-extra=--field-kernel-cache --v36-fixes";;
    s2) A="--prod-dir ${QI2_ROOT}/LADrs32 --prod-tag member2_snapshot_2100 --prod-json-name member2_snapshot_2100.json"; X="--both-extra=--field-kernel-cache --v36-fixes --nk-jvp exact";;
  esac
  until awk '{exit !($1 < 90)}' /proc/loadavg; do sleep 120; done
  echo "gateC3_$M launch $(date -u +%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/status.txt
  cd $C && nice -n 10 $PY gate_regression_v2b.py $A "$X" --v36 $C/engine_twofield_v3.py --out $R/gateC3_$M --records 3 --engine-dir $E > $R/gateC3_$M.log 2>&1
  echo "gateC3_$M exit $? $(date -u +%H:%M:%S)" >> $R/status.txt
done
