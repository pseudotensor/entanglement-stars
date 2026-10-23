#!/bin/bash
# resolve_fd_v36b_pod1_v1.sh (solve builder 9, analyst; engine spec 51.1 (3)): the cause's readout. engine_collapse_v36b WITHOUT --nk-jvp exact
# (the Newton direction by scipy's finite difference of F, as v32's) and so without --pair-workers, every other argument the v32
# continuation's, from the same upcast T/r_s 16 slice to T/r_s 16.875 (7 records). Converging at 16.875: the exact direction chose the side;
# stalling: the transport or the histories did. A label, nothing waits on it.
set -u
D=${QI2_ROOT}/v32fd; mkdir -p $D && cd $D || exit 1
T0=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b; TAG=${T0}_v36bc_fd16875
[ -e collapse_$TAG.json ] && mv collapse_$TAG.json collapse_$TAG.json.guardstop.$(date +%s)
IN=${QI2_ROOT}/in
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=${QI2_ROOT}/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date -u '+%F %T UTC') slice md5 $(md5sum $IN/orbitals_${T0}_T0016.00_c128.npz | cut -c1-32)" >> status.txt
nice -n 10 python ${QI2_ROOT}/engine_collapse_v36b.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 16.875 --prep-ell 4 --tau 8 --gpu 0 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --field-kernel-cache --v36-fixes --resume-slice $IN/orbitals_${T0}_T0016.00_c128.npz --resume-run $IN/run/$T0 --out . --tag $TAG > $TAG.log 2>&1
rc=$?; echo "engine exit $rc $(date -u '+%F %T UTC')" >> status.txt; exit $rc
