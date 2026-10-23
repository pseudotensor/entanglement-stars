#!/bin/bash
# Strand 2b timing test on R4 (one case, no enclosure or uniqueness) and the 2a transport addendum; 2 threads in total.
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p timing1
( /usr/bin/time -v nice -n 10 $PY code/strand2b_row_R4.py --out timing1 --block 2,1 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k 1 --no-enclosure --no-unique > timing1/case.log 2>&1; echo "exit $?" >> timing1/case.log ) &
( cd ${QI2_ROOT}/s2a_gate && mkdir -p addendum1 && nice -n 10 $PY code/strand2a_addendum_transport.py --out addendum1 > addendum1/addendum.log 2>&1; echo "exit $?" >> addendum1/addendum.log ) &
wait
