#!/bin/bash
# Strand 4 Part B.1 and B.2 (strand4_partB1_v1.py) on the stored executed-star states (the paper lane's profiles, copied with md5 into
# partB_inputs/); CPU, 2 threads.
set -u
D=${QI2_ROOT}/s4_controls
cd $D || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p partB_run1
nice -n 10 ${QI2_ROOT}/python code/strand4_partB1_v1.py --inputs $D/partB_inputs --out partB_run1 > partB_run1/partB1.log 2>&1
echo "exit $?" >> partB_run1/partB1.log
