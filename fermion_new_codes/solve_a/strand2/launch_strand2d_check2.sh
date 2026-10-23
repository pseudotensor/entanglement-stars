#!/bin/bash
# Strand 2d library checks, parts (a) and (b) on CPU (strand2d_check_v2.py; part (c) needs a free GPU and runs separately); 2 threads.
set -u
D=${QI2_ROOT}/s2d_jet
cd $D || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nice -n 10 ${QI2_ROOT}/python code/strand2d_check_v2.py --out check2 --device cpu --parts a,b > check2.log 2>&1
echo "exit $?" >> check2.log
