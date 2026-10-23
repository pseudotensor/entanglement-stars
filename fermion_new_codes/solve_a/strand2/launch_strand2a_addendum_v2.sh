#!/bin/bash
# Strand 2a addendum items A.1c and A.4 (one-body; 4 threads).
cd ${QI2_ROOT}/s2a_gate || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p addendum2
nice -n 10 ${QI2_ROOT}/python code/strand2a_addendum_transport_v2.py --out addendum2 > addendum2/addendum.log 2>&1
echo "exit $?" >> addendum2/addendum.log
