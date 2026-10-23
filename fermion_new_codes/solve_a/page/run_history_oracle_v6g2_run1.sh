#!/bin/bash
# the history oracle v6g2 on the box CPU (builder analyst): page_history_oracle_v6g2.py (the floor criterion of the partners' orthogonality; the dense oracle of page_history_v6g with both
# ledgers, the cubic run, v6f against v6g, the merged history, the follow with partial files, the done marker, a changed record, two
# channels); nice 10, 8 threads
cd ${QI2_ROOT}/v6
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
date -u +"start %H:%M:%S UTC" > history_oracle_v6g2_run1.marker
nice -n 10 ${QI2_ROOT}/python page_history_oracle_v6g2.py --out runs/history_oracle_v6g2_run1 --device cpu --threads 8 > history_oracle_v6g2_run1.log 2>&1
echo "exit $? $(date -u +%H:%M:%S) UTC" >> history_oracle_v6g2_run1.marker
