#!/bin/bash
# Builder cost probe of the C8 2b route (strand2b_c8_cost_v1.py): one build, one F and one Jacobian per block at the flat point.  One thread.
cd ${QI2_ROOT}/s2b_row || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p c8cost1
nice -n 10 ${QI2_ROOT}/python code/strand2b_c8_cost_v1.py --out c8cost1 > c8cost1/cost.log 2>&1
echo "exit $?" >> c8cost1/cost.log
