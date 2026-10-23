#!/bin/bash
# Computation C, gate C (the regression) on the box CPU (RUN_ORDER_C_v1.md section 3, the box fallback): the busi lane's H100 (pod 3)
# is held by a process stalled on its network volume, and the box GPUs belong to the solve lane. One member per call:
#   gateC_box_v1.sh s4   (the sigma/a 4 member, C3lad32, engine_collapse_v31 plain)
#   gateC_box_v1.sh s2   (the sigma/a 2 member, LADrs32 snapshot, engine_collapse_v32 newton)
# The solve lane's modules are a read-only copy in this lane's own compC_engine (engine/code/*.py and page/v6/page_ops_v5.py, md5-listed in
# compC_engine/MD5.txt), so v35 runs with its working directory in this lane's space.
M=$1; R=${QI2_ROOT}/compC; C=${QI2_ROOT}/compC; mkdir -p $R
E=${QI2_ROOT}/compC_engine
export BUSI_ENGINE_DIR=$E BUSI_PAGE_DIR=$E PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=${QI2_ROOT}/python
case $M in
  s4) A="--prod-dir ${QI2_ROOT}/C3lad32 --prod-tag c31_rs32_A_R2_tau2_ell4_prep4_avg";;
  s2) A="--prod-dir ${QI2_ROOT}/LADrs32 --prod-tag member2_snapshot_2100 --prod-json-name member2_snapshot_2100.json";;
  *) echo "usage: $0 s4|s2"; exit 2;;
esac
echo "gateC_$M launch $(date -u +%H:%M:%S) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/status.txt
cd $C && nice -n 10 $PY gate_regression_v1.py $A --out $R/gateC_$M --records 3 --engine-dir $E > $R/gateC_$M.log 2>&1
echo "gateC_$M exit $? $(date -u +%H:%M:%S)" >> $R/status.txt
