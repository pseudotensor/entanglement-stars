#!/bin/bash
# The economy engine's validation (SPEC_final4b_engine.md 18.5) on the H100 pod, beside the running v15 primary (the GPU idles while the
# primary's CPU constraint solves run): r_s 64, ell 4, tau 8, R2, kappa_sea 10, reading A, the v9 primary's arguments, T/rs 0.5 (5 records):
# run 1 the strict rule (1e-9, fields 1e-8) point sampled; run 2 the lead's rule (1e-7, 1e-6) point sampled; run 3 the lead's rule kernel
# averaged; then the CUDA-streams benchmark (engine_eigh_streams_bench_v1) on run 1's last record. Sequential, 4 threads each.
# Usage (on the pod, detached): cd ${QI2_ROOT}/engine; nohup bash code/launch_h100_val17_v1.sh > logs_h100_val17_launch.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=/usr/bin/python3.11
mkdir -p C3val
COMMON="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --kappa-sea 10 --reference R2 --T-over-rs 0.5 --out C3val --gpu 0"
for spec in "val17_rs64_strict_dense|--stop-E 1e-9 --stop-fields 1e-8 --site-fields dense" \
            "val17_rs64_default_dense|--site-fields dense" \
            "val17_rs64_default_avg|--site-fields averaged"; do
  tag=${spec%%|*}; args=${spec#*|}
  echo "$tag start $(date -u +%H:%M:%S) UTC"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $PY code/engine_collapse_v17.py $COMMON $args --tag $tag > logs_h100_$tag.log 2>&1 < /dev/null
  echo "$tag exit $? $(date -u +%H:%M:%S) UTC"
done
echo "streams bench start $(date -u +%H:%M:%S) UTC"
OMP_NUM_THREADS=4 $PY code/engine_eigh_streams_bench_v1.py --fields C3val/collapse_val17_rs64_strict_dense_fields.npz --record -1 --kmax 10 --m 0.2 --gpu 0 --reps 2 --md C3val/streams_bench_rs64_v1.md > logs_h100_streams_bench_rs64.log 2>&1 < /dev/null
echo "streams bench exit $? $(date -u +%H:%M:%S) UTC"
echo "DONE $(date -u +%H:%M:%S) UTC"
