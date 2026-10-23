#!/bin/bash
# Engine job V3D (SPEC_final4b_engine.md 2.4): engine_validate3d_v1.py at L = 11 and 15, s = 3 and 4, B = 0 and 0.3, both centrifugal
# placements (R_n = n + 1, n + 1/2), m a = 0.3, A = 0.3, counts 16, 48, 120; one GPU, sequential, nice 10, 1 thread.  Builder analyst.
GPU=${1:-0}
cd ${QI2_ROOT}/engine || exit 1
mkdir -p V3D
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for L in 11 15; do for s in 3 4; do for B in 0 0.3; do for ho in 0 1; do
  tag=L${L}_s${s}_B${B}_ho${ho}
  echo "V3D $tag start $(date -u +%H:%M:%S) UTC (gpu $GPU)" >> status.txt
  nice -n 10 ${QI2_ROOT}/python code/engine_validate3d_v1.py --L $L --s $s --B $B --half-offset $ho --gpu $GPU --out V3D --tag $tag > V3D/$tag.log 2>&1
  echo "V3D $tag exit $? $(date -u +%H:%M:%S)" >> status.txt
done; done; done; done
echo "V3D DONE" >> status.txt
