#!/bin/bash
# Task 4 of the rewritten goal (the single fermion's per-normal cut coefficient against m a), runnable before the lead's specification:
# strandS_sphere_v6.py's per-ray inner form re-assembled on the saved chain values of the four tilted normals not yet re-assembled
# ((1,1,1), (2,1,0), (2,1,1), (2,2,1); S3v3's runs, 17 masses each: 0 and the 16 values aM sqrt(x) of the record's ladders), outputs in
# ${QI2_ROOT}/.  CPU, 10 workers, nice 10, sequential.  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p ../t4b
PY=${QI2_ROOT}/python
for n in 111 210 211 221; do
  until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 6 ]; do sleep 30; done
  date -u "+t4b eta_v6 $n start %H:%M:%S UTC (10 workers)" >> ../t4b/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/strandS_sphere_v6.py --from-json S3v3/sphere_$n.json \
    --from-npz S3v3/sphere_${n}_points.npz --workers 10 --out-json ../t4b/sphere_${n}_v6base.json > ../t4b/run_eta_v6_$n.log 2>&1
  echo "t4b eta_v6 $n exit $? $(date -u +%H:%M:%S)" >> ../t4b/status.txt
done
cp S3v6/sphere_100_v6base.json S3v6/sphere_110_v6base.json ../t4b/
echo DONE >> ../t4b/status.txt
