#!/bin/bash
# resolve_16125_tight_pod1_v1.sh (solve builder 8, analyst; engine spec 50.1 (2)): the continuation's first record re-solved under the tightened
# rule. The same upcast T/r_s 16 slice and run copy as launch_v32cont_pod2_v1.sh (md5-equal copies), the same engine and arguments
# (engine_collapse_v36b --pair-workers 2 at the pod 1 member's arguments) with --stop-fields 1e-07 and --stop-E 1e-08 in place of the stored
# 1e-06 and 1e-07, ending after the record 16.125 (--segment-end-T-over-rs 16.125). The box compares its 16.125 record with the stopped member's
# stored one and with the continuation's (each root separately, the kept residuals beside). Run as pod1_queue_v1's placed job.
set -u
D=${QI2_ROOT}/v32tight; cd $D || exit 1
T0=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b; TAG=${T0}_v36bc_tight16125
[ -e collapse_$TAG.json ] && { echo "refusing: collapse_$TAG.json exists"; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=${QI2_ROOT}/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date -u '+%F %T UTC') slice md5 $(md5sum in/orbitals_${T0}_T0016.00_c128.npz | cut -c1-32)" >> status.txt
nice -n 10 python ${QI2_ROOT}/engine_collapse_v36b.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --field-kernel-cache --v36-fixes --pair-workers 2 --stop-fields 1e-07 --stop-E 1e-08 --resume-slice $D/in/orbitals_${T0}_T0016.00_c128.npz --resume-run $D/in/run/$T0 --segment-end-T-over-rs 16.125 --out . --tag $TAG > $TAG.log 2>&1
rc=$?; echo "engine exit $rc $(date -u '+%F %T UTC')" >> status.txt; exit $rc
