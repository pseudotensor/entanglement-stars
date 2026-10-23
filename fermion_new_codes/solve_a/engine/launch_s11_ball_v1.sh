#!/bin/bash
# launch_s11_ball_v1.sh (builder analyst; S11 of the quasi-frozen ruling, owner-approved (inbox #36); brief lead/BRIEF_D_S11_builder.md):
# the self-consistent massless collapse pathfinder at r_s/a 16, G/a^2 1/4, m 0, ell 2a, tau 2, on engine_collapse_v33 (version 32's solve:
# Newton-Krylov with 40 Krylov vectors and 2 restarts, the halving line search, the two-stage source continuation, the end above 1e-3;
# the Weyl-certified reference transport of ops v3; engine_fields_v17, N' = 0 on an empty interior), the massless Fermi ball of radius
# 4 r_s from the chiral Slater seeds (6 in channel 1, x0 = R0/2, sigma = R0/4, the inward boost tuned after the constraint solve to
# E a = 32), 1024 sites (64 r_s: the chain of the Page recursion), 10 channels, 320 records (T/r_s 40), the S11 readouts and the stop
# where M(r) < 0 outside the central averaging cell; complex128 slices at T/r_s 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40.
# Usage (from ${QI2_ROOT}/engine on the box): bash code/launch_s11_ball_v1.sh <gpu index>   (the lane's placement)
# The r_s/a 32 run at ell 2a and 4a is not launched by this file (the owner's decision after MASSLESS_ball_pathfinder_v1.md).
set -u
GPU=${1:?the GPU index}
cd ${QI2_ROOT}/engine || exit 2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4   # 4 CPU threads beside the GPU (the field solves are on the CPU); the builder's total stays at 16 with D on 12
PY=${QI2_ROOT}/python
TAG=s11_ball_rs16_ell2_tau2_v1
mkdir -p S11ball states/$TAG
echo "start $(date -u '+%Y-%m-%d %H:%M:%S UTC') gpu $GPU pid $$" >> S11ball/status_$TAG.txt
nice -n 10 $PY code/engine_collapse_v33.py --m 0 --g 0.25 --rs 16 --nsites 1024 --ell 2 --kappa-sea 10 --tau 2 --T-over-rs 40 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --slices-c128 0,1,2,3,4,5,6,8,10,12,16,20,24,28,32,36,40 --slice-dir states/$TAG --gpu $GPU --out S11ball --tag $TAG > S11ball/$TAG.log 2>&1
echo "exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> S11ball/status_$TAG.txt
