#!/bin/bash
# Strand S job S3 on strandS_sphere_v3.py (the lead's sec. 9.5 rulings; the panels aligned with the disc boundary): [100] first (the cross-check
# against S2, which version 1 failed), then 111, 110, 210, 211, 221; aM = 0.4, 0.2, 0.1, 0.05, both rectangles; 10 CPU workers, one normal at a
# time, each start gated on the lane's busy processes (at most 15 before the pool's 10).  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p S3v3
for n in 1,0,0 1,1,1 1,1,0 2,1,0 2,1,1 2,2,1; do
  tag=$(echo $n | tr -d ,)
  until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 6 ]; do sleep 30; done
  date -u "+S3v3 $tag start %H:%M:%S UTC (10 workers)" >> S3v3/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strandS_sphere_v3.py \
    --normal $n --workers 10 --out-json S3v3/sphere_$tag.json > S3v3/run_$tag.log 2>&1
  echo "S3v3 $tag exit $? $(date -u +%H:%M:%S)" >> S3v3/status.txt
done
echo DONE >> S3v3/status.txt
