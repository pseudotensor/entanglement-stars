#!/bin/bash
# Strand 3 addendum part B' step 2 and the part A refinement (strand3_Bprime2_v1.py), 12 single-thread workers.  Output s3_entropy/Bprime2.
set -u
D=${QI2_ROOT}/s3_entropy
cd $D || exit 1
mkdir -p Bprime2
date -u '+start %Y-%m-%d %H:%M:%S UTC' > Bprime2/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand3_Bprime2_v1.py --out Bprime2 --workers 12 > Bprime2/run.log 2>&1
echo "exit $?" >> Bprime2/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> Bprime2/status.txt
echo DONE >> Bprime2/status.txt
