#!/bin/bash
# Strand 6 (h) (SPEC_strand6_validation.md, final3 (h)): strand6h_vertex_v1.py on strand 6's two-thread chain after (i).  Builder analyst.
D=${QI2_ROOT}/s6_val
mkdir -p $D/s6h_run1
cd $D || exit 1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > s6h_run1/status.txt
until grep -q DONE s6i_run1/status.txt 2>/dev/null; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> s6h_run1/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand6h_vertex_v1.py \
  --md s6h_run1/RESULT_s6h_tables_v1.md --out-json s6h_run1/s6h_vertex.json > s6h_run1/run.log 2>&1
echo "vertex exit $? $(date -u +%H:%M:%S)" >> s6h_run1/status.txt
echo DONE >> s6h_run1/status.txt
