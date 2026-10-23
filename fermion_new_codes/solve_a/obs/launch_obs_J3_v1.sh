#!/bin/bash
# Observables job J3 (SPEC_final4b_observables.md sec. 3): obs_raymap_pg_v1.py on the exact Painleve-Gullstrand radial profile (control),
# r_s = 64, R_m = 16, N = 8192, one process per channel on one GPU, nice 10, 1 thread: m in {0, 1/512, 1/256} x j in {1/2, 3/2, 5/2} at
# sign -1, then (0, 1/2, +1) as the time-reversal pair.  (r_s = 128 at N = 16384 does not fit one 48 GB GPU with the dense route: the
# matrix of 32768^2 complex entries alone is 17 GB; told to the lead.)  Builder analyst.
GPU=${1:-1}
cd ${QI2_ROOT}/obs || exit 1
mkdir -p J3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for ch in 0:0.5:-1 0.001953125:0.5:-1 0.00390625:0.5:-1 0:1.5:-1 0.001953125:1.5:-1 0.00390625:1.5:-1 0:2.5:-1 0.001953125:2.5:-1 0.00390625:2.5:-1 0:0.5:1; do
  tag=rs64_$(echo $ch | tr ':' '_' | sed 's/-1$/m1/; s/_1$/_p1/')
  echo "J3 $tag start $(date -u +%H:%M:%S) UTC (gpu $GPU)" >> status.txt
  nice -n 10 ${QI2_ROOT}/python code/obs_raymap_pg_v1.py --rs 64 --Rm 16 --nsites 8192 --channels $ch --gpu $GPU --outdir J3 --tag $tag > J3/$tag.log 2>&1
  echo "J3 $tag exit $? $(date -u +%H:%M:%S)" >> status.txt
done
echo "J3 DONE" >> status.txt
