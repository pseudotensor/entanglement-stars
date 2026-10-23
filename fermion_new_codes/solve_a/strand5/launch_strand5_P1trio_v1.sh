#!/bin/bash
# Strand 5 P1 addendum: strand5_P1_trio_v1.py (well under a second, one thread), started inside S7a's smoke window (the S7a chain runs one
# single-thread smoke job before its six parallel jobs, so five of its six threads are idle then).  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
until grep -q "^start" ${QI2_ROOT}/status.txt 2>/dev/null; do sleep 5; done
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_P1_trio_v1.py \
  --family-new P1run1/lstar_m0.01_aH0.1_local_ka0_g0_0_pk_p1new.json --s2-new P1run1/stage2_lstar_m0.01_aH0.1_local_ka0_g0_0_pk_p1new.json \
  --md P1run1/RESULT_P1_trio_v1.md > P1run1/trio.log 2>&1
echo "P1 trio exit $? $(date -u +%H:%M:%S)" >> P1run1/status.txt
