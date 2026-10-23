#!/bin/bash
# Strand 2c, version 2 of the script (GPU dtype fix): a GPU-against-CPU check at L = 4, then L = 16 on GPU 0 and L = 8, 12 on
# GPU 2 (one process per GPU; both free at the batch launch: GPU 1 is another user's).
set -u
D=${QI2_ROOT}/s2c_mismatch
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/run2
mkdir -p $OUT/check_gpu $OUT/check_cpu
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
nice -n 10 $PY code/strand2c_mismatch_v2.py --out $OUT/check_gpu --L 4 --gpu 0 --Hs 1 > $OUT/check_gpu.log 2>&1; echo "check gpu exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand2c_mismatch_v2.py --out $OUT/check_cpu --L 4 --gpu -1 --Hs 1 > $OUT/check_cpu.log 2>&1; echo "check cpu exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand2c_mismatch_v2.py --out $OUT --L 16 --gpu 0 > $OUT/L16.log 2>&1 &
(nice -n 10 $PY code/strand2c_mismatch_v2.py --out $OUT --L 8 --gpu 2 > $OUT/L8.log 2>&1; echo "L8 exit $?" >> $OUT/status.txt; nice -n 10 $PY code/strand2c_mismatch_v2.py --out $OUT --L 12 --gpu 2 > $OUT/L12.log 2>&1; echo "L12 exit $?" >> $OUT/status.txt) &
wait
echo "L16: $(tail -1 $OUT/L16.log)" >> $OUT/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
