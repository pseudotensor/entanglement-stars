#!/bin/bash
# PERF RD-1c v9c7 on the H3 segment (busi): run_seg_v9c6.sh's arguments word for word (R2's c64s8L production line from the T 13.00 slice
# d17763e7 to 13.1875, 6 records; the four v9c2 PERF options; H3's A1 and C1 in every arm), on the box GPU given as $2 (never 0: a GPU that
# nvidia-smi showed free), nice 10, host threads 8. One arm per call, sequential:
#   G6   engine_twofield_v9c6 A1+C1 (the reference of the default)            G7   v9c7 A1+C1 (the default: bitwise G6 on the same device)
#   T12  v9c6 A1+C1 + SYM-C1 + SYM-C2 + --light-walk tabulated (RD-4's arm)   H12  v9c7 T12 + --heavy-eigh banded (the option's box smoke)
# Usage. run_seg_v9c7.sh ARM GPU
ARM=$1; GPU=${2:?gpu}
[ "$GPU" = 0 ] && { echo "STOP: never cuda:0"; exit 2; }
H=${QI2_ROOT}/H3_run; W=${QI2_ROOT}/seg; mkdir -p $W
ST=${QI2_ROOT}/v9c7
export CUDA_VISIBLE_DEVICES=$GPU BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
ENG=$ST/engine_twofield_v9c7.py; V6=$ST/engine_twofield_v9c6.py
BASE="--perf-h3-band-gather --perf-h3-weight-gather"; SYM="--perf-sym-light-sector --perf-sym-ladder-sector"
case $ARM in G6) ENG=$V6; X="";; G7) X="";; T12) ENG=$V6; X="$SYM --light-walk tabulated";;
  H12) X="$SYM --light-walk tabulated --heavy-eigh banded";; *) echo "unknown arm $ARM"; exit 2;; esac
S=${QI2_ROOT}/orbitals_C64_s8_ell8_tau2_twofield_v4L_T0013.00.npz
ON="--perf-eig-cache-gib 4 --perf-dens-memo on --perf-uhx-cache-gib 2 --perf-writer on"
O=$W/$ARM; rm -rf $O; mkdir -p $O/states
( cd $H/snap && diff -q <(md5sum collapse_* | sort -k2) <(sort -k2 box_md5.txt) > /dev/null ) || { echo "STOP: the snapshot differs from snap/box_md5.txt"; exit 2; }
echo "arm $ARM gpu $GPU ($(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader -i $GPU)) threads $OMP_NUM_THREADS; engine $(basename $ENG) md5 $(md5sum < $ENG | cut -c1-32); options '$BASE $X'; slice md5 $(md5sum < $S | cut -c1-8)"
echo "start $(date -u +%FT%TZ)"; t0=$(date +%s)
nice -n 10 $PY $ENG --gpu 0 --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 56 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --reference-solve lagged --cut-transport sector --light-max-pieces 8192 --slices-c128 '' --slices-light-c128 '' --ell 8 \
 --slice-dir $O/states --out $O --tag C64_s8_ell8_tau2_twofield_v4L --resume-slice $S --resume-run $H/snap/C64_s8_ell8_tau2_twofield_v4L --segment-end-T-over-rs 13.1875 --anchor-passage lz $ON $BASE $X > $O/engine_run.log 2>&1
rc=$?
echo "end $(date -u +%FT%TZ) rc $rc wall $(( $(date +%s) - t0 )) s"
