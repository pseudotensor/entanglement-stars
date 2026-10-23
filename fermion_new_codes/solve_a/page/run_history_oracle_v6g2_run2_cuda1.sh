#!/bin/bash
# the history oracle v6g2 on box GPU 1 (builder analyst): page_history_oracle_v6g2.py on the device, the CUDA paths of page_history_v6g (a few hundred MB beside S11), nice 10, 8 threads
cd ${QI2_ROOT}/v6
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
date -u +"start %H:%M:%S UTC" > history_oracle_v6g2_run2_cuda1.marker
nice -n 10 ${QI2_ROOT}/python page_history_oracle_v6g2.py --out runs/history_oracle_v6g2_run2_cuda1 --device cuda:1 --threads 8 > history_oracle_v6g2_run2_cuda1.log 2>&1
echo "exit $? $(date -u +%H:%M:%S) UTC" >> history_oracle_v6g2_run2_cuda1.marker
