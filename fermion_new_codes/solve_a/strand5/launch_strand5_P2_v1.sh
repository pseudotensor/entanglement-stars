#!/bin/bash
# Strand 5 production P2 (SPEC_strand5_covariant_row.md sec. 3): the family table from J1's record and J4run4's (0.01, 0.1) local_ka0 families at
# (0, 0) and (0.2, 1); no solve; 2 threads (the production chain).  Waits for F'''(d)'s report (thread budget).  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
mkdir -p P2run1
until grep -q "^report exit" ${QI2_ROOT}/status.txt; do sleep 20; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' > P2run1/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_P2_table_v1.py \
  --j1 J1run3/star_v4.json \
  --lat "local_ka0 (0.01, 0.1) couplings (0, 0)=J4run4/lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json,local_ka0 (0.01, 0.1) couplings (0.2, 1)=J4run4/lstar_m0.01_aH0.1_local_ka0_g0.2_1_pk.json" \
  --md P2run1/RESULT_P2_tables_v1.md > P2run1/run.log 2>&1
echo "P2 exit $? $(date -u +%H:%M:%S)" >> P2run1/status.txt
