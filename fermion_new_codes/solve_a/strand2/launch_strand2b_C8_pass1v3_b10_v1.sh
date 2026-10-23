#!/bin/bash
# Strand 2b C8 pass 1 on library v3 (batch v2 -> driver v3: the common-sea Schur overlap, the projector rank fixed at the flat start), block
# (1,0): 60 cases, no enclosure or uniqueness, 12 single-thread workers.  The library v1 run of the same cases (c8pass1/) is kept as the route
# cross-check.
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/c8pass1v3
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+start b10 %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
nice -n 10 $PY code/strand2b_batch_C8_v2.py --out $OUT --workers 12 --blocks "1,0" --no-enclosure --no-unique > $OUT/batch_b10.log 2>&1; echo "b10 batch exit $?" >> $OUT/status.txt
date -u '+end b10 %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
