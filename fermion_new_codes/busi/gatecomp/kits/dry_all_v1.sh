#!/bin/bash
# dry_all_v1.sh (busi, GATECOMP kits, #400): the CPU dry runs of every launch line on the box (parse + check modes of prep_dry_v1.py at the
# full-size args; no evolution, no GPU, no input file read), then each line's parsed args diffed against its reference run's args:
# CV2048 against job (b)'s json, the members against each other, the controls against W256_s8. Box: nice 10, 8 threads.
set -u
K=${QI2_ROOT}/kits; CODE=${CODE:-${QI2_ROOT}/code16d}; ENG=engine_collapse_v36nq_rd13mf.py
PY=${QI2_ROOT}/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
echo "engine $(md5sum < $CODE/$ENG | cut -c1-32)  fields $(md5sum < $CODE/engine_fields_v16.py | cut -c1-32)  code dir $CODE"
for f in $K/lines/*.args; do
  n=$(basename $f .args); D=$K/dry/$n; rm -rf $D; mkdir -p $D
  for m in parse check; do
    nice -n 10 $PY $K/code/prep_dry_v1.py $m $CODE $ENG $f $D > $D/$m.log 2>&1; rc=$?
    echo "$n $m rc $rc: $(tail -1 $D/$m.log | cut -c1-160)"
  done
done
