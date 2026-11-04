#!/bin/bash
# busi reading formation_newton_v1: the runner (box only; nice 10; 8 threads; python -I; logs under OUT/logs). fn_md5_v1.sh runs first.
O=${QI2_ROOT}/formation_newton_v1; C=$O/code; PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 TORCH_NUM_THREADS=8
mkdir -p $O/logs
for s in fn_records_v1 fn_fields_v1; do
  echo "[$(date -u +%FT%TZ)] $s start" | tee -a $O/logs/run.log
  nice -n 10 $PY -I $C/$s.py > $O/logs/$s.log 2>&1; echo "[$(date -u +%FT%TZ)] $s rc $?" | tee -a $O/logs/run.log
done
