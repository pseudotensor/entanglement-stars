#!/bin/bash
# launch_v36member_pod1_v3.sh (the lane's analyst builder; engine spec 45.1 (2), 45.3 (1)): the decisive tolerance check. The sigma/a 2 ladder
# member re-solved from T = 0 on engine_collapse_v36 --field-kernel-cache --v36-fixes --nk-jvp exact with launch_v36member_pod1_v2.sh's arguments
# unchanged except the Newton stop rule's two tolerances, tightened by a factor 10 through the engine's own options:
#   --stop-fields 1e-07 (stored value 1e-06, the engine default the member's line used: the kept residual's max-norm tolerance)
#   --stop-E      1e-08 (stored value 1e-07, the engine default the member's line used: the confirming step's |dE/E| tolerance)
# (the other tolerances, --field-rtol 1e-12 and --field-atol 1e-16 of the field integration and --end-above 1e-3, are not the stop rule and
# stay as stored). nice 10, 4 threads, expandable segments (allocation only), ${QI2_ROOT}/v36member_r2, beside A2. The test through T/r_s 1.25 against
# the stopped v36 run at the floor (v36member_switch_pod1_v3.sh), then the member's run to 40 or its end rule with 45.2's labels.
D=${QI2_ROOT}/v36member_r2; mkdir -p $D && cd $D || exit 1
T=c32_rs64_A_R2_tau8_ell2_prep4_avg_v36tol10
[ -e collapse_$T.json ] && { echo "refusing: collapse_$T.json exists"; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=${QI2_ROOT}/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "v36member v3 (tolerances tightened tenfold: --stop-fields 1e-07 from 1e-06, --stop-E 1e-08 from 1e-07) start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> status.txt
nice -n 10 python ${QI2_ROOT}/engine_collapse_v36.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/$T --nk-jvp exact --field-kernel-cache --v36-fixes --stop-fields 1e-07 --stop-E 1e-08 --out . --tag $T > $T.log 2>&1
echo "v36member v3 exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
