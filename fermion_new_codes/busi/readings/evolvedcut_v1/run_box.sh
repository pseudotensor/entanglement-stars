#!/bin/bash
# run_box.sh (busi lane, cold check of EVOLVEDCUT_route_compare_v1): the three readers on the box, CPU, nice 10, 8 threads.
# Scripts are copied to ${QI2_ROOT}/code and run from there; outputs beside them in OUT.
set -e
PY=${QI2_ROOT}/python
OUT=${QI2_ROOT}/evolvedcut_v1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd $OUT/code
case "$1" in
  arith)   nice -n 10 $PY ec_arith.py ;;
  spectra) nice -n 10 $PY ec_spectra.py ;;
  recompute) nohup nice -n 10 $PY ec_recompute.py > $OUT/recompute.stdout 2>&1 < /dev/null & echo "launched pid $!" ;;
  *) echo "usage: run_box.sh arith|spectra|recompute" ;;
esac
