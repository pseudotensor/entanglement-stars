#!/bin/bash
# launch_v36member_pod1_v2.sh (the lane's analyst builder; engine spec 44.1 (1)): launch_v36member_pod1_v1.sh's arguments unchanged (the sigma/a 2
# ladder member re-solved from T = 0 on engine_collapse_v36 --field-kernel-cache --v36-fixes --nk-jvp exact; nice 10, 4 threads; ${QI2_ROOT}/v36member),
# with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (allocation only), beside A2 after the v35 jvp member's stop. Version 1's out-of-memory
# run's files are moved to ${QI2_ROOT}/oom_run1 first (kept as a record). The gate: v36member_switch_pod1_v2.sh.
D=${QI2_ROOT}/v36member; mkdir -p $D && cd $D || exit 1
T=c32_rs64_A_R2_tau8_ell2_prep4_avg_v36
if [ -e collapse_$T.json ] || [ -d states ]; then mkdir -p oom_run1 && mv -n collapse_$T* $T.log gate_v36_vs_v35jvp.* switch.log states oom_run1/ 2>/dev/null; fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=${QI2_ROOT}/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "v36member v2 start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> status.txt
nice -n 10 python ${QI2_ROOT}/engine_collapse_v36.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/$T --nk-jvp exact --field-kernel-cache --v36-fixes --out . --tag $T > $T.log 2>&1
echo "v36member v2 exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
