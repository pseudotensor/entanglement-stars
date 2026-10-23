#!/bin/bash
# Spec 19.2: the ell 8 averaged control of the r_s 64 production on the H100 (the ell control changes ell alone): engine_collapse_v17, r_s 64,
# ell 8, tau 8, R2, kappa_sea 10, reading A, 2048 sites, T/rs 40, site fields averaged, the lead's stop rule. 4 host threads.
# Usage (on the pod, detached): cd ${QI2_ROOT}/engine; nohup bash code/launch_h100_ell8_17_v1.sh > logs_h100_ell8_17_launch.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p C3prod
tag=c17_rs64_A_R2_tau8_ell8_avg
echo "$tag start $(date -u +%H:%M:%S) UTC"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /usr/bin/python3.11 code/engine_collapse_v17.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A \
  --nsites 2048 --ell 8 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --site-fields averaged --out C3prod --gpu 0 --tag $tag > logs_h100_$tag.log 2>&1 < /dev/null
echo "$tag exit $? $(date -u +%H:%M:%S) UTC"
