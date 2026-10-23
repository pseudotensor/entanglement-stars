#!/bin/bash
# PERF H3 v9c4 on the H3_run segment (busi, owner rule #334): R2's c64s8L production line (run_h3_v9c2_cpu_pyspy_r6.sh's arguments, without
# py-spy) from the T 13.00 slice d17763e7 to 13.1875 (6 records) on the box CPU, nice 10, H1 thread limits OMP/MKL/OPENBLAS 70 (nproc 72 - 2),
# CUDA hidden, the four v9c2 PERF options on as in H3_run. One arm per call, sequential for the timing:
#   REF  engine_twofield_v9c3c (the bitwise reference; not timed),  OFF/OFF2  v9c4 at the defaults,  A1  --perf-h3-band-gather,
#   B8  --perf-h3-anchor-workers 8,  AB8  both.
# Usage. run_h3seg_v9c4.sh ARM
ARM=$1
H=${QI2_ROOT}/H3_run; W=${QI2_ROOT}/seg; mkdir -p $W
ST=${QI2_ROOT}/v9c4
export CUDA_VISIBLE_DEVICES="" BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=70 MKL_NUM_THREADS=70 OPENBLAS_NUM_THREADS=70 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
ENG=$ST/engine_twofield_v9c4.py
case $ARM in REF) ENG=$ST/engine_twofield_v9c3c.py; H3="";; OFF|OFF2) H3="";; A1) H3="--perf-h3-band-gather";; B8) H3="--perf-h3-anchor-workers 8";;
  AB8) H3="--perf-h3-band-gather --perf-h3-anchor-workers 8";; *) echo "unknown arm $ARM"; exit 2;; esac
S=${QI2_ROOT}/orbitals_C64_s8_ell8_tau2_twofield_v4L_T0013.00.npz
ON="--perf-eig-cache-gib 4 --perf-dens-memo on --perf-uhx-cache-gib 2 --perf-writer on"
O=$W/$ARM; rm -rf $O; mkdir -p $O/states
( cd $H/snap && diff -q <(md5sum collapse_* | sort -k2) <(sort -k2 box_md5.txt) > /dev/null ) || { echo "STOP: the snapshot differs from snap/box_md5.txt"; exit 2; }
echo "arm $ARM threads OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS; engine $(basename $ENG) md5 $(md5sum < $ENG | cut -c1-32); options '$H3'; slice md5 $(md5sum < $S | cut -c1-8)"
echo "start $(date -u +%FT%TZ)"; t0=$(date +%s)
nice -n 10 $PY $ENG --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 56 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --reference-solve lagged --cut-transport sector --light-max-pieces 8192 --slices-c128 '' --slices-light-c128 '' --ell 8 \
 --slice-dir $O/states --out $O --tag C64_s8_ell8_tau2_twofield_v4L --resume-slice $S --resume-run $H/snap/C64_s8_ell8_tau2_twofield_v4L --segment-end-T-over-rs 13.1875 --anchor-passage lz $ON $H3 > $O/engine_run.log 2>&1
rc=$?
echo "end $(date -u +%FT%TZ) rc $rc wall $(( $(date +%s) - t0 )) s"
