#!/bin/bash
# launch_v36bm_pod1_v1.sh (solve builder 7, analyst; engine spec 48.1 (4), 48.5 (1), 48.4 (3)): the sigma/a 2 member's second history at the
# stored tolerances. engine_collapse_v36b.py --pair-workers 2 (bitwise against v36 with the option off and on, results/engine/V36b/) with the
# stopped v36 run's exact arguments (launch_v36member_pod1_v2.sh: --field-kernel-cache --v36-fixes --nk-jvp exact, the engine's stored stop rule
# --stop-fields 1e-06 and --stop-E 1e-07 by default, --end-above 1e-3) plus --slices-c128 every 1 r_s from 1 to 40 (the c64 list's clocks are
# all in it, where the engine writes complex128), in ${QI2_ROOT}/v36b_member beside A2, nice 10, 4 threads.
# Fresh start: bash launch_v36bm_pod1_v1.sh            -> tag c32_rs64_A_R2_tau8_ell2_prep4_avg_v36b
# Restart:     bash launch_v36bm_pod1_v1.sh SLICE RUN N -> the slice restart of v34 (a) (--resume-slice SLICE --resume-run RUN), tag ..._v36b_rN
# The current tag is written to CURRENT_TAG (read by the guard, the switch and the box mirror).
D=${QI2_ROOT}/v36b_member; mkdir -p $D && cd $D || exit 1
T0=c32_rs64_A_R2_tau8_ell2_prep4_avg_v36b
if [ -n "${1:-}" ]; then T=${T0}_r${3:?}; RES="--resume-slice $1 --resume-run $2"; else T=$T0; RES=""; fi
[ -e collapse_$T.json ] && { echo "refusing: collapse_$T.json exists"; exit 1; }
echo $T > CURRENT_TAG
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=${QI2_ROOT}/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SL=$(seq -s, 1 40)
echo "v36bm start tag $T ${RES:-from T = 0} $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> status.txt
nice -n 10 python ${QI2_ROOT}/engine_collapse_v36b.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --slices-c128 $SL --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/$T --nk-jvp exact --field-kernel-cache --v36-fixes --pair-workers 2 $RES --out . --tag $T > $T.log 2>&1
echo "v36bm exit $? tag $T $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
