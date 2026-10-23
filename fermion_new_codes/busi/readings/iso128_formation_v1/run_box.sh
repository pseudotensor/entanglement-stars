#!/bin/bash
# run_box.sh: run the reading's scripts on the box (CPU, nice 10, 8 threads). Outputs under ${QI2_ROOT}/.
set -u
PY=${QI2_ROOT}/python
OUT=${QI2_ROOT}/iso128_formation_v1
CODE=$OUT/code
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
mkdir -p $OUT/logs
cd $CODE
for s in "$@"; do
  echo "== $s $(date -u +%FT%TZ)"
  nice -n 10 $PY -I $CODE/$s.py > $OUT/logs/$s.log 2>&1
  echo "rc $? $(date -u +%FT%TZ)"
  tail -c 1500 $OUT/logs/$s.log
done
