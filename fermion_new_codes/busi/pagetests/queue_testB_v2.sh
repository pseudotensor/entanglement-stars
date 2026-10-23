#!/bin/bash
# Test B queue 2 (after queue 1): the slow-deceleration completion control (Delta x 4) at r_s/a 32 and 64.
set -u
cd ${QI2_ROOT}/code
PY=${QI2_ROOT}/python
OUT=${QI2_ROOT}/testB
LOG=${QI2_ROOT}/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
until grep -q "queue done" $LOG/queue_testB_v1.log 2>/dev/null; do sleep 60; done
run() { tag=$1; shift; echo "$(date -u +%FT%TZ) start $tag" >> $LOG/queue_testB_v2.log
  nice -n 10 $PY pt_testB_run_v1.py "$@" --out $OUT --tag $tag --device cuda:1 > $LOG/$tag.log 2>&1
  echo "$(date -u +%FT%TZ) end $tag rc $?" >> $LOG/queue_testB_v2.log; }
run B_rs32_Aslow --rs 32 --delta-factor 2.0
run B_rs64_Aslow --rs 64 --delta-factor 2.0
echo "$(date -u +%FT%TZ) queue done" >> $LOG/queue_testB_v2.log
