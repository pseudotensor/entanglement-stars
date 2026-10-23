#!/bin/bash
# Strand S job S3 (SPEC_strandS_graded_entropy.md sec. 4; the builder's plan put to the lead in turn 10, labelled, pending the lead's ruling):
# the per-normal graded planar coefficient over the paper's mesh-2 normal set, [100] first (the cross-check against S2), at aM = 0.4, 0.2,
# 0.1, 0.05, both rectangles; the inner form below gap_0 = 0.05 (L cap 480); one corner x 4 and the inversion, checked pointwise.
# 12 CPU workers, one normal at a time, nice.  Timing test on (2,1,1): about 1.7 CPU-hours per normal.  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p S3
for n in 1,0,0 1,1,1 1,1,0 2,1,0 2,1,1 2,2,1; do
  tag=$(echo $n | tr -d ,)
  date -u "+S3 $tag start %H:%M:%S UTC (12 workers)" >> S3/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strandS_sphere_v1.py \
    --normal $n --workers 12 --out-json S3/sphere_$tag.json > S3/run_$tag.log 2>&1
  echo "S3 $tag exit $? $(date -u +%H:%M:%S)" >> S3/status.txt
done
echo DONE >> S3/status.txt
