#!/bin/bash
# launch_v36d_relaunch_pod4_v1.sh (solve builder 11, analyst; engine spec 53.1 (2), (3)): the sigma/a 2 point relaunched on
# engine_collapse_v36d --nk-set resolve from the v36c run's own T/r_s 17 complex128 slice (md5 bde4c81f..., the box's /data mirror) with the
# run's exact arguments (launch_v36cc_pod2_v1.sh's line, --pair-workers 2 included), complex128 slices every 1 r_s, mirrored to /data by the
# box's mirror_slices_v2 instance v36d_relaunch. Python: ${QI2_ROOT}/venv_v36d (numpy 2.4.6, scipy 1.17.1, torch 2.4.1: pod 2's versions, so
# that the records 17.125 to 17.5 can be bitwise against the run). The card's primary on pod 4, never stopped by the queue's guard.
# The gate is the relaunch itself (gate_v36d_relaunch_box_v1.sh on the box): 17.125 to 17.5 against the run's records (bitwise expected),
# 17.625 to 18.125 against the v32 member's at twice the floor.
# Fresh:   bash launch_v36d_relaunch_pod4_v1.sh               -> tag ..._nk40r2b_v36cc_v36d
# Restart: bash launch_v36d_relaunch_pod4_v1.sh SLICE RUN N   -> tag ..._nk40r2b_v36cc_v36d_rN from its own complex128 slice
E=${QI2_ROOT}/engine; D=$E/v36d_relaunch; mkdir -p $D && cd $D || exit 1
R=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc; T0=${R}_v36d
if [ -n "${1:-}" ]; then T=${T0}_r${3:?}; SLC=$1; RES="--resume-slice $1 --resume-run $2"
else T=$T0; SLC=$D/in/orbitals_${R}_T0017.00.npz; RES="--resume-slice $SLC --resume-run $D/in/run/$R"; fi
[ -e collapse_$T.json ] && { echo "refusing: collapse_$T.json exists"; exit 1; }
echo $T > CURRENT_TAG
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=$E/code_v36d PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SL=$(seq -s, 17 40)
echo "v36d start tag $T $RES $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$ slice md5 $(md5sum $SLC | cut -c1-32)" >> status.txt
nice -n 10 ${QI2_ROOT}/python $E/code_v36d/engine_collapse_v36d.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c64 20,25,30,40 --slices-c128 $SL --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/$T --nk-jvp exact --field-kernel-cache --v36-fixes --pair-workers 2 --nk-set resolve $RES --out . --tag $T > $T.log 2>&1
echo "v36d exit $? tag $T $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
