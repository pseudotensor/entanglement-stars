#!/bin/bash
# launch_v32cont_pod2_v1.sh (solve builder 8, analyst; engine spec 49.3 (1), route (a); the owner: "if it can be started on new solver, do
# it. Ensure any restart states are copied over"): the v32 sigma/a 2 member (tag c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b, stopped by the lead
# at its T/r_s 17.5 record) continued on engine_collapse_v36b --pair-workers 2 from its latest mirrored complex64 slice (T/r_s 16), made
# complex128 and re-orthonormalized by slice_c64_to_c128_v1.py (Gram deviation printed there), with v34 (a)'s slice restart. Labelled "a
# continuation within the member's solver spread". Arguments: the member's own (its json args) plus v36b's --nk-jvp exact --field-kernel-cache
# --v36-fixes --pair-workers 2 (as the pod 1 v36b member), complex128 slices every 1 r_s from 17 to 40 (the c64 list's later clocks are in it).
# Its first 2 records (16.125, 16.25) are compared with the member's stored ones at twice the floor, masks equal, by the box's
# gate_v32cont_box_v1.sh; on PASS the run keeps going as the member's continuation.
# Fresh:   bash launch_v32cont_pod2_v1.sh                 -> from the converted T/r_s 16 slice, tag ..._nk40r2b_v36bc
# Restart: bash launch_v32cont_pod2_v1.sh SLICE RUN N     -> tag ..._nk40r2b_v36bc_rN from its own complex128 slice
# The current tag is written to CURRENT_TAG.
D=${QI2_ROOT}/v32cont; mkdir -p $D && cd $D || exit 1
T0=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36bc
if [ -n "${1:-}" ]; then T=${T0}_r${3:?}; RES="--resume-slice $1 --resume-run $2"
else T=$T0; RES="--resume-slice $D/in/orbitals_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_T0016.00_c128.npz --resume-run $D/in/run/c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b"; fi
[ -e collapse_$T.json ] && { echo "refusing: collapse_$T.json exists"; exit 1; }
echo $T > CURRENT_TAG
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=${QI2_ROOT}/code_v36b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SL=$(seq -s, 17 40)
echo "v32cont start tag $T $RES $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> status.txt
nice -n 10 python ${QI2_ROOT}/engine_collapse_v36b.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c64 20,25,30,40 --slices-c128 $SL --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/$T --nk-jvp exact --field-kernel-cache --v36-fixes --pair-workers 2 $RES --out . --tag $T > $T.log 2>&1
echo "v32cont exit $? tag $T $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
