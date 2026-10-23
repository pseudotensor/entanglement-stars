#!/bin/bash
# the history oracle on page_history_v6k (option (v) off) and the hold test (engine spec 43.3 (1)); box CPU, nice 10, 8 threads each, one after the other
cd ${QI2_ROOT}/v6
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=${QI2_ROOT}/python
echo "start $(date -u +%H:%M:%S) UTC" > v6k_tests.marker
nice -n 10 $PY page_history_oracle_v6k.py --out runs/history_oracle_v6k_run1 --device cpu --threads 8 > history_oracle_v6k_run1.log 2>&1
echo "oracle exit $? $(date -u +%H:%M:%S) UTC" >> v6k_tests.marker
nice -n 10 $PY compare_oracle_v6k_v1.py runs/history_oracle_v6j_run1 runs/history_oracle_v6k_run1 > compare_oracle_v6k_v1.txt 2>&1
echo "compare exit $? $(date -u +%H:%M:%S) UTC" >> v6k_tests.marker
nice -n 10 $PY holdtest_v6k_v1.py --out runs/holdtest_v6k_run1 --device cpu --threads 8 --engine-dir ${QI2_ROOT}/code > holdtest_v6k_run1.log 2>&1
echo "holdtest exit $? $(date -u +%H:%M:%S) UTC" >> v6k_tests.marker
