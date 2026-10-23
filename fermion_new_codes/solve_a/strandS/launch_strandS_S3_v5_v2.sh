#!/bin/bash
# Strand S, S3 controls (the lead's judgment sec. 8.5(6)) on strandS_sphere_v5.py, version 2 of the queue: after 111_ext (started by version 1
# of this launcher, left running), the |p|-rule control at (1,1,1) comes first so that it can decide whether the tilted normals' extensions
# need the rule; then the other normals' extensions and the matching-radius variants.  CPU, 10 workers, nice 10.  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p S3v5
run() { local tag=$1; shift
  until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 6 ]; do sleep 30; done
  date -u "+S3v5 $tag start %H:%M:%S UTC (10 workers)" >> S3v5/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strandS_sphere_v5.py "$@" \
    --workers 10 --out-json S3v5/sphere_$tag.json > S3v5/run_$tag.log 2>&1
  echo "S3v5 $tag exit $? $(date -u +%H:%M:%S)" >> S3v5/status.txt; }
until [ -s S3v5/sphere_111_ext.json ] && ! ps -u jon -o args= | grep -q "[s]trandS_sphere_v5.py --normal 1,1,1 --aM 0.025,0.0125"; do sleep 30; done
echo "S3v5 111_ext ended, its JSON present $(date -u +%H:%M:%S) (launcher version 1 was stopped while it ran)" >> S3v5/status.txt
run 111_prule --normal 1,1,1 --aM 0.4,0.2,0.1,0.05,0.025,0.0125 --pnorm-rule --massive-check 12
for n in 1,0,0 1,1,0 2,1,0 2,1,1 2,2,1; do run $(echo $n | tr -d ,)_ext --normal $n --aM 0.025,0.0125; done
run 111_glim0.03 --normal 1,1,1 --aM 0.4,0.2,0.1,0.05,0.025,0.0125 --Lcap 800
run 111_glim0.08 --normal 1,1,1 --aM 0.4,0.2,0.1,0.05,0.025,0.0125 --Lcap 300
echo DONE >> S3v5/status.txt
