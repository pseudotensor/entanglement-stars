#!/bin/bash
# Strand S job S2 (SPEC_strandS_graded_entropy.md secs. 1 to 3, 5 to 7; the lead's hand-back job 4): the flat [100] graded cut on S1's table,
# both rectangles at the 11 spacings 1.6 to 0.0025, hat k_0 by J2e's two routes, controls C1, C2, C6, C7, C8, the extrapolations on aM <= 0.1.
# 8 CPU workers, nice.  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p S2
date -u "+S2 start %H:%M:%S UTC (8 workers)" >> S2/status.txt
STRAND5_CODE=${QI2_ROOT}/code OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 \
  ${QI2_ROOT}/python code/strandS_flat_v3.py --table S1/strandS_table_v1.json \
  --aM 1.6,0.8,0.4,0.2,0.1,0.05,0.025,0.0125,0.00625,0.005,0.0025 --workers 8 --out-json S2/strandS_flat_v3.json --md S2/RESULT_S2_flat.md > S2/run.log 2>&1
echo "S2 exit $? $(date -u +%H:%M:%S)" >> S2/status.txt
echo DONE >> S2/status.txt
