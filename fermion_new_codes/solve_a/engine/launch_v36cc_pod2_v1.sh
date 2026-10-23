#!/bin/bash
# launch_v36cc_pod2_v1.sh (solve builder 9, analyst; engine spec 51.1 (2), (4)): the sigma/a 2 point's run: the v32 member's history to
# T/r_s 16 continued on engine_collapse_v36c --nk-set resolve --pair-workers 2 at the pod 1 member's arguments (as launch_v32cont_pod2_v1.sh)
# from the upcast complex128 T/r_s 16 slice, complex128 slices every 1 r_s from 17 to 40 (mirrored to /data by the box's mirror_slices),
# labelled "continued on v36c (--nk-set resolve)". Its records 16.125 to 17.5 are read against the stopped v32 member's stored records at twice
# the floor (compare_member_floor_v2 --criterion twice) as they land. Beside the pod 2 queue's job (the placed process under the queue's 8192
# MiB guard); this run is never stopped by the guard.
# Fresh:   bash launch_v36cc_pod2_v1.sh                 -> tag ..._nk40r2b_v36cc
# Restart: bash launch_v36cc_pod2_v1.sh SLICE RUN N     -> tag ..._nk40r2b_v36cc_rN from its own complex128 slice
E=${QI2_ROOT}/engine; D=$E/v32cont_v36c; mkdir -p $D && cd $D || exit 1
T0=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc
if [ -n "${1:-}" ]; then T=${T0}_r${3:?}; RES="--resume-slice $1 --resume-run $2"
else T=$T0; RES="--resume-slice $E/v32cont/in/orbitals_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_T0016.00_c128.npz --resume-run $E/v32cont/in/run/c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b"; fi
[ -e collapse_$T.json ] && { echo "refusing: collapse_$T.json exists"; exit 1; }
echo $T > CURRENT_TAG
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=$E/code_v36c PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SL=$(seq -s, 17 40)
echo "v36cc start tag $T $RES $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> status.txt
nice -n 10 python $E/code_v36c/engine_collapse_v36c.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c64 20,25,30,40 --slices-c128 $SL --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/$T --nk-jvp exact --field-kernel-cache --v36-fixes --pair-workers 2 --nk-set resolve $RES --out . --tag $T > $T.log 2>&1
echo "v36cc exit $? tag $T $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
