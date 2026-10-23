#!/bin/bash
# resolve_17_tight_pod1_v1.sh (solve builder 10, analyst; engine spec 52.1 (3)): the v36c run's records 17.125 to 17.5 re-solved from the
# run's own T/r_s 17 complex128 slice (md5-equal copy of pod 2's, via the box mirror) with the run's engine and arguments
# (engine_collapse_v36c --nk-set resolve --pair-workers 2, launch_v36cc_pod2_v1.sh's line) and --stop-fields 1e-07 --stop-E 1e-08 in place
# of the stored 1e-06 and 1e-07, ending after 17.5 (--segment-end-T-over-rs 17.5). Read against the run's records (its own residual's share)
# and the stopped member's (within twice the floor confirms the reading, else that record is a term of the seam). Run as pod1_queue_v1b's job.
set -u
D=${QI2_ROOT}/v36ctight17; cd $D || exit 1
T0=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b; R=${T0}_v36cc; TAG=${T0}_v36cc_tight17
[ -e collapse_$TAG.json ] && mv collapse_$TAG.json collapse_$TAG.json.guardstop.$(date +%s)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=${QI2_ROOT}/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date -u '+%F %T UTC') slice md5 $(md5sum in/orbitals_${R}_T0017.00.npz | cut -c1-32)" >> status.txt
nice -n 10 python ${QI2_ROOT}/engine_collapse_v36c.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --field-kernel-cache --v36-fixes --pair-workers 2 --nk-set resolve --stop-fields 1e-07 --stop-E 1e-08 --resume-slice $D/in/orbitals_${R}_T0017.00.npz --resume-run $D/in/run/$R --segment-end-T-over-rs 17.5 --out . --tag $TAG > $TAG.log 2>&1
rc=$?; echo "engine exit $rc $(date -u '+%F %T UTC')" >> status.txt; exit $rc
