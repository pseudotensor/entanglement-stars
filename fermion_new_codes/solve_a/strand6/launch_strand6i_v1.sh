#!/bin/bash
# Strand 6 (i) (SPEC_strand6_validation.md, final3 (i)): the quadrature route (strand6i_kernel_sign_v1.py) on strand 6's two-thread chain, which
# starts when J5decomp3 ends (thread budget: J5 slots A and B 6 each, the production chain 2, this chain 2).  Builder analyst.
D=${QI2_ROOT}/s6_val
mkdir -p $D/s6i_run1 $D/code
cd $D || exit 1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > s6i_run1/status.txt
until grep -q DONE ${QI2_ROOT}/status.txt; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> s6i_run1/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand6i_kernel_sign_v1.py \
  --md s6i_run1/RESULT_s6i_tables_v1.md --out-json s6i_run1/s6i_quadrature.json > s6i_run1/run.log 2>&1
echo "quadrature exit $? $(date -u +%H:%M:%S)" >> s6i_run1/status.txt
echo DONE >> s6i_run1/status.txt
