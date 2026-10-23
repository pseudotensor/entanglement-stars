#!/bin/bash
# Strand 0 smoke test (part A --quick) on the box; builder check of the code before the full run.
set -u
D=${QI2_ROOT}/s0_source_allocation
cd $D || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
date -u '+start %Y-%m-%d %H:%M:%S UTC' > smoke2_status.txt
nice -n 10 $PY code/strand0_partA_data_v2.py --quick --out smoke2 --stored-jobs ${QI2_ROOT}/native_jobs.tar.gz --stored-kscan ${QI2_ROOT}/native_kscan.json.gz > smoke2.log 2>&1
echo "exit $?" >> smoke2_status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> smoke2_status.txt
echo DONE >> smoke2_status.txt
