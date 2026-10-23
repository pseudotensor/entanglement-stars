#!/bin/bash
# Strand S job S1 (SPEC_strandS_graded_entropy.md sec. 2, the lead's hand-back job 3): the 2-component chain table mu in [0.006, 3] at 200 per
# decade, muL >= 24, L <= 4096, nfft = 4L; the ladder at 6 masses on GPU 1 (one process; GPU 1 carries only another user's idle litellm
# contexts, as for K1); the fixed-alpha fit on [0.006, 0.05], C3 and C4.  10 CPU workers, nice.  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p S1
date -u "+S1 start %H:%M:%S UTC (10 workers, ladder on cuda:1)" >> S1/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strandS_table_v1.py \
  --mu-min 0.006 --mu-max 3 --per-decade 200 --muL 24 --L-cap 4096 --fit-hi 0.05 --ladder 0.006,0.01,0.03,0.1,0.3,1.0 \
  --c3 0.05,0.1,0.3,1.0,2.0 --gpu 1 --workers 10 --out-json S1/strandS_table_v1.json --md S1/RESULT_S1_table.md > S1/run.log 2>&1
echo "S1 exit $? $(date -u +%H:%M:%S)" >> S1/status.txt
echo DONE >> S1/status.txt
