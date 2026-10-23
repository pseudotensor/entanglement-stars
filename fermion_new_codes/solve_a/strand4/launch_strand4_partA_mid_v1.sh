#!/bin/bash
# Strand 4 Part A rerun (the lead's ruling after Part A, item 2): the primary reading m0 = 0.06741313 with the dictionary 'p2mid' (p = -2
# quartics, the executed code's midpoint hopping weight Nbar/psibar^2) at L = 6 and 8 (strand4_partA_v2.py); one thread, sequential.
set -u
D=${QI2_ROOT}/s4_controls
cd $D || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FULL_NUMERICS=$D/full_numerics
OUT=$D/partA_run2
mkdir -p $OUT
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
for L in 6 8; do
  nice -n 10 ${QI2_ROOT}/python code/strand4_partA_v2.py --out $OUT --L $L --m0 0.06741313 --dict p2mid > $OUT/partA_L$L.log 2>&1
  echo "L$L m0.06741313 p2mid exit $?" >> $OUT/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
