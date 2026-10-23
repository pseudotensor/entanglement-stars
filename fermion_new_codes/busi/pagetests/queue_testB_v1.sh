#!/bin/bash
# Test B queue on the box GPU 1 (inbox #428): the ladder r_s/a 32, 64, 128 (completion A, R_bf 0.5 r_s), completion B (R_bf 0.3 r_s) at 64,
# and the collapse-step control (dt halved) at 32.
set -u
cd ${QI2_ROOT}/code
PY=${QI2_ROOT}/python
OUT=${QI2_ROOT}/testB
LOG=${QI2_ROOT}/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
run() { # tag args...
  tag=$1; shift
  echo "$(date -u +%FT%TZ) start $tag" >> $LOG/queue_testB_v1.log
  nice -n 10 $PY pt_testB_run_v1.py "$@" --out $OUT --tag $tag --device cuda:1 > $LOG/$tag.log 2>&1
  echo "$(date -u +%FT%TZ) end $tag rc $?" >> $LOG/queue_testB_v1.log
}
run B_rs32_A --rs 32
run B_rs64_A --rs 64
run B_rs128_A --rs 128
run B_rs64_B --rs 64 --Rbf-rs 0.3
run B_rs32_A_dthalf --rs 32 --dt-collapse-rs 0.0078125 --no-raymap
echo "$(date -u +%FT%TZ) queue done" >> $LOG/queue_testB_v1.log
