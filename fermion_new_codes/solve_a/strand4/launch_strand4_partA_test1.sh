#!/bin/bash
# Strand 4 Part A, first test: L = 6, m0 = 0.06741313 (the executed star's flat-matched bare mass) with dictionary p2, one thread.
set -u
D=${QI2_ROOT}/s4_controls
cd $D || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FULL_NUMERICS=$D/full_numerics
mkdir -p test1
nice -n 10 ${QI2_ROOT}/python code/strand4_partA_v1.py --out test1 --L 6 --m0 0.06741313 --dict p2 > test1/partA.log 2>&1
echo "exit $?" >> test1/partA.log
