#!/bin/bash
# Strand 2b C8 pass 1, block (1,0): every law case (3 m x 4 coupling rows x 5 k = 60) through strand2b_row_C8_v2.solve_case (the sec. 10
# route), without the enclosure and the uniqueness scan (their scope is the lead's ruling).  12 single-thread workers.  The (2,0) block
# follows in the same output folder once its cost is known.
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/c8pass1
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+start b10 %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
nice -n 10 $PY code/strand2b_batch_C8_v1.py --out $OUT --workers 12 --blocks "1,0" --no-enclosure --no-unique > $OUT/batch_b10.log 2>&1; echo "b10 batch exit $?" >> $OUT/status.txt
date -u '+end b10 %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
