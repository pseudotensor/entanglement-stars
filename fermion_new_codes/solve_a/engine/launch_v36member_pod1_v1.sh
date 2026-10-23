#!/bin/bash
# launch_v36member_pod1_v1.sh (the lane's analyst builder; engine spec 43.2 (1)): the sigma/a 2 ladder member re-solved from T = 0 on
# engine_collapse_v36 --field-kernel-cache --v36-fixes --nk-jvp exact, the jvp member's exact arguments otherwise, beside the jvp member and A2
# on pod 1's H100; nice 10, 4 threads; ${QI2_ROOT}/v36member. The gate and the switch: v36member_switch_pod1_v1.sh.
D=${QI2_ROOT}/v36member; mkdir -p $D && cd $D || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=${QI2_ROOT}/code
T=c32_rs64_A_R2_tau8_ell2_prep4_avg_v36
echo "v36member start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> status.txt
nice -n 10 python ${QI2_ROOT}/engine_collapse_v36.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/$T --nk-jvp exact --field-kernel-cache --v36-fixes --out . --tag $T > $T.log 2>&1
echo "v36member exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
