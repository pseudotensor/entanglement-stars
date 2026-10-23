#!/bin/bash
# launch_jvpmember_pod1_v2.sh (the lane's analyst builder; engine spec 44.1 (2)): the FAIL branch only: the v35 jvp member relaunched from T = 0 with
# pod1_h100_queue_v2.sh's arguments unchanged, in ${QI2_ROOT}/jvpmember_r2 (the stopped copy's records in ${QI2_ROOT}/jvpmember kept as the record);
# nice 10, 4 threads, expandable segments (allocation only).
D=${QI2_ROOT}/jvpmember_r2; mkdir -p $D && cd $D || exit 1
T=c32_rs64_A_R2_tau8_ell2_prep4_avg_v35jvp
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=${QI2_ROOT}/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "jvp member r2 start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> status.txt
nice -n 10 python ${QI2_ROOT}/engine_collapse_v35.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/$T --nk-jvp exact --out . --tag $T > $T.log 2>&1
echo "jvp member r2 exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
