#!/bin/bash
# Engine job V3D, the L = 15 cases on engine_validate3d_v2.py (sparse density products; identical sums to version 1), skipping any case
# already on record (L15_s3_B0_ho0 finished on version 1).  One GPU, sequential, nice 10, 1 thread.  Builder analyst.
GPU=${1:-0}
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for L in 15; do for s in 3 4; do for B in 0 0.3; do for ho in 0 1; do
  tag=L${L}_s${s}_B${B}_ho${ho}
  [ -f V3D/v3d_$tag.json ] && continue
  echo "V3D v2 $tag start $(date -u +%H:%M:%S) UTC (gpu $GPU)" >> status.txt
  nice -n 10 ${QI2_ROOT}/python code/engine_validate3d_v2.py --L $L --s $s --B $B --half-offset $ho --gpu $GPU --out V3D --tag $tag > V3D/$tag.log 2>&1
  echo "V3D v2 $tag exit $? $(date -u +%H:%M:%S)" >> status.txt
done; done; done; done
echo "V3D v2 DONE $(date -u +%H:%M:%S) UTC" >> status.txt
