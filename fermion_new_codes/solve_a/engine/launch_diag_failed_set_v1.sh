#!/bin/bash
# Diagnostic for the lead's ruling on the r_s 32 set's two stopped runs: the same configurations (the S-shell 3.875 to 4.125 r_s; reading B
# at R_0 = 16 r_s) on engine_collapse_v16 (version 9's Picard solve, Anderson beta 0.5, point sampled) on the box CPU for T/rs 1 (4 steps),
# to separate the initial-value solve and the economy iteration from the configuration itself. 12 threads, nice 10, sequential.
# Usage (on the box, detached): cd ${QI2_ROOT}/engine; nohup bash code/launch_diag_failed_set_v1.sh > C3diag/launch.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
mkdir -p C3diag
C="--m 0.2 --g 0.25 --rs 32 --nsites 1024 --T-over-rs 1 --ell 4 --tau 8 --kappa-sea 10 --reference R2 --out C3diag"
for spec in "diag16_rs32_A_R2_Sshell|--reading A --R0-over-rs 4 --shell 3.875,4.125" "diag16_rs32_B_R2_R0x16|--reading B --R0-over-rs 16"; do
  tag=${spec%%|*}; args=${spec#*|}
  echo "$tag start $(date -u +%H:%M:%S) UTC"
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 nice -n 10 $PY code/engine_collapse_v16.py $C $args --tag $tag > C3diag/$tag.log 2>&1 < /dev/null
  echo "$tag exit $? $(date -u +%H:%M:%S) UTC"
done
echo "DONE $(date -u +%H:%M:%S) UTC"
