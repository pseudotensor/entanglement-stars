#!/bin/bash
# the history oracle v6i (builder analyst): page_history_oracle_v6i.py on page_history_v6i.py (the ledgers' partners on the pure state,
# the orthonormal archive and their Gram brought to the identity), device cpu on the box; nice 10, 8 threads
cd ${QI2_ROOT}/v6
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
date -u +"start %H:%M:%S UTC" > history_oracle_v6i_run1.marker
nice -n 10 ${QI2_ROOT}/python page_history_oracle_v6i.py --out runs/history_oracle_v6i_run1 --device cpu --threads 8 > history_oracle_v6i_run1.log 2>&1
echo "exit $? $(date -u +%H:%M:%S) UTC" >> history_oracle_v6i_run1.marker
