#!/bin/bash
# engine_collapse_v17's CPU tests at r_s 16 (512 sites, ell 4, tau 8, T/rs 2; the arguments of the t12..t15 equivalence runs in C3eq/):
# the strict rule (1e-9, fields 1e-8) and the lead's rule (1e-7, 1e-6), point sampled, the lead's rule kernel averaged, and the strict rule
# with Anderson mixing; verification, delta_rtol every 2nd record and M1's Simpson check exercised. nice 10, 12 threads, sequential.
# Usage (on the box, detached): cd ${QI2_ROOT}/engine; nohup bash code/launch_t17_cpu_tests_v1.sh > C3eq/v17/launch.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
mkdir -p C3eq/v17
COMMON="--rs 16 --nsites 512 --ell 4 --tau 8 --kappa-sea 10 --reference R2 --T-over-rs 2 --out C3eq/v17 --verify-every 2 --delta-rtol-every 2"
for spec in "t17_rs16_strict_dense|--stop-E 1e-9 --stop-fields 1e-8 --site-fields dense" \
            "t17_rs16_default_dense|--site-fields dense" \
            "t17_rs16_default_avg|--site-fields averaged" \
            "t17_rs16_strict_dense_anderson|--stop-E 1e-9 --stop-fields 1e-8 --site-fields dense --mix anderson"; do
  tag=${spec%%|*}; args=${spec#*|}
  echo "$tag start $(date -u +%H:%M:%S) UTC"
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 nice -n 10 $PY code/engine_collapse_v17.py $COMMON $args --tag $tag > C3eq/v17/$tag.log 2>&1 < /dev/null
  echo "$tag exit $? $(date -u +%H:%M:%S) UTC"
done
echo "DONE $(date -u +%H:%M:%S) UTC"
