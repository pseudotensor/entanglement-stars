#!/bin/bash
# Strand 4 Part B.3 to B.5 (strand4_partB3_v1.py) on the stored executed-star states, queued in the lane's two-thread slot behind the
# 2b Jacobian-gap check (s2b_row/jacgap1).
set -u
D=${QI2_ROOT}/s4_controls
cd $D || exit 1
until grep -q "^exit" ${QI2_ROOT}/jacgap.log 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p partB_run1
nice -n 10 ${QI2_ROOT}/python code/strand4_partB3_v1.py --inputs $D/partB_inputs --out partB_run1 > partB_run1/partB3.log 2>&1
echo "exit $?" >> partB_run1/partB3.log
