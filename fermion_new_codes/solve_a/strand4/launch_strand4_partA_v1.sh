#!/bin/bash
# Strand 4 Part A production (strand4_partA_v1.py): L in {6, 8} x m0 in {0.06741313 (the executed star's flat-matched bare mass),
# 0.3 (the spec's literal value)} x dictionary in {spec (s^-6, (s s)^-3), p2 (the executed star's p = -2)}; 8 single-thread runs,
# 4 at a time.  The lead's readings (INBOX) select which rows are read.
set -u
D=${QI2_ROOT}/s4_controls
cd $D || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FULL_NUMERICS=$D/full_numerics
OUT=$D/partA_run1
mkdir -p $OUT
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
for L in 6 8; do for m0 in 0.06741313 0.3; do for dct in spec p2; do echo "$L $m0 $dct"; done; done; done > $OUT/cases.txt
cat $OUT/cases.txt | xargs -P 4 -L 1 bash -c 'nice -n 10 ${QI2_ROOT}/python code/strand4_partA_v1.py --out '"$OUT"' --L $0 --m0 $1 --dict $2 > '"$OUT"'/partA_L$0_m$1_$2.log 2>&1; echo "L$0 m$1 $2 exit $?" >> '"$OUT"'/status.txt'
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
