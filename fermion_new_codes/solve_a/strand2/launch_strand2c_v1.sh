#!/bin/bash
# Strand 2c on the box (spec solve/specs/SPEC_strand2c_mismatch.md), without the P2 profiles (they need 2b's solutions; a later
# call adds them).  One process per GPU: GPU 0 takes L = 16; GPU 2 takes L = 8 then L = 12.  CPU threads 2 per process.
set -u
D=${QI2_ROOT}/s2c_mismatch
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/run1
mkdir -p $OUT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
nice -n 10 $PY code/strand2c_mismatch.py --out $OUT --L 16 --gpu 0 > $OUT/L16.log 2>&1 &
(nice -n 10 $PY code/strand2c_mismatch.py --out $OUT --L 8 --gpu 2 > $OUT/L8.log 2>&1; echo "L8 exit $?" >> $OUT/status.txt; nice -n 10 $PY code/strand2c_mismatch.py --out $OUT --L 12 --gpu 2 > $OUT/L12.log 2>&1; echo "L12 exit $?" >> $OUT/status.txt) &
wait
echo "L16: $(tail -1 $OUT/L16.log)" >> $OUT/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
