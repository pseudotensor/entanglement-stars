#!/bin/bash
# run_s8end_v1.sh (busi, inbox #457): box CPU, nice 10, 8 threads; the zero-card reads on F256_s8's end files.
set -e
O=${QI2_ROOT}/s8end_reads; C=$O/code
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd $O
nice -n 10 ${QI2_ROOT}/python $C/s8end_reads_v1.py $O/s8end_reads_v1.json \
  ${QI2_ROOT}/F256_s8_end_T5.00 ${QI2_ROOT}/run \
  ${QI2_ROOT}/run_stop_backup ${QI2_ROOT}/code \
  ${QI2_ROOT}/code16d_flat ${QI2_ROOT}/stop_flat_h100f_v2.py > $O/s8end_reads_v1.out 2>&1
echo done >> $O/s8end_reads_v1.out
