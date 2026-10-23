#!/bin/bash
# Strand S on the flat [100] cut, the full run (strandS_flat_v1.py, the builder's plan pending the lead's specification): the chain table
# from mu = 0.015 to 4 at 40 per decade (L = 12/mu, the doubled L as the error), the asymptotic fit on [0.015, 0.06], the node-adapted transverse
# quadrature, the rectangles (0, 1/2, 1/2, 1) and (1, 2, 2, 3), aM from 1.6 to 0.00625; 8 CPU workers, nice.  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p run1
date -u "+run1 start %H:%M:%S UTC (8 workers)" >> run1/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strandS_flat_v1.py \
  --mu-lo 0.015 --mu-hi 4 --mu-per-decade 40 --mu-fit-hi 0.06 --Lf 12 --workers 8 --out-json run1/strandS_flat_v1.json --md run1/RESULT_strandS_flat_v1.md > run1/run.log 2>&1
echo "run1 exit $? $(date -u +%H:%M:%S)" >> run1/status.txt
echo DONE >> run1/status.txt
