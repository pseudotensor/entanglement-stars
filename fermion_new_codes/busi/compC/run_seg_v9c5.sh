#!/bin/bash
# PERF SYM-C1/SYM-C2 v9c5 on the H3 segment (busi): R2's c64s8L production line (run_h3seg_v9c4.sh's arguments) from the T 13.00 slice
# d17763e7 to 13.1875 (6 records), on the box GPU given as $2 (an RTX A6000 that nvidia-smi showed free), nice 10, host threads 16, the four
# v9c2 PERF options on as in H3_run, H3's A1 and C1 on in every arm (the solve lane's card baseline). One arm per call, sequential for the timing:
#   G4   engine_twofield_v9c4 A1+C1 (the reference; the card A/B's A arm)
#   G5   v9c5 A1+C1 (the options off: bitwise G4 on the same device)
#   S1   v9c5 A1+C1 + --perf-sym-light-sector       S2  v9c5 A1+C1 + --perf-sym-ladder-sector
#   S12  v9c5 A1+C1 + both (the card A/B's B arm)
# Usage. run_seg_v9c5.sh ARM GPU
ARM=$1; GPU=${2:-0}
H=${QI2_ROOT}/H3_run; W=${QI2_ROOT}/seg; mkdir -p $W
ST=${QI2_ROOT}/v9c5
export CUDA_VISIBLE_DEVICES=$GPU BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
PY=${QI2_ROOT}/python
ENG=$ST/engine_twofield_v9c5.py
BASE="--perf-h3-band-gather --perf-h3-weight-gather"
case $ARM in G4) ENG=$ST/engine_twofield_v9c4.py; X="";; G5) X="";; S1) X="--perf-sym-light-sector";; S2) X="--perf-sym-ladder-sector";;
  S12) X="--perf-sym-light-sector --perf-sym-ladder-sector";; *) echo "unknown arm $ARM"; exit 2;; esac
S=${QI2_ROOT}/orbitals_C64_s8_ell8_tau2_twofield_v4L_T0013.00.npz
ON="--perf-eig-cache-gib 4 --perf-dens-memo on --perf-uhx-cache-gib 2 --perf-writer on"
O=$W/$ARM; rm -rf $O; mkdir -p $O/states
( cd $H/snap && diff -q <(md5sum collapse_* | sort -k2) <(sort -k2 box_md5.txt) > /dev/null ) || { echo "STOP: the snapshot differs from snap/box_md5.txt"; exit 2; }
echo "arm $ARM gpu $GPU ($(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader -i $GPU)) threads $OMP_NUM_THREADS; engine $(basename $ENG) md5 $(md5sum < $ENG | cut -c1-32); options '$BASE $X'; slice md5 $(md5sum < $S | cut -c1-8)"
echo "start $(date -u +%FT%TZ)"; t0=$(date +%s)
nice -n 10 $PY $ENG --gpu 0 --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau 2 --T-over-rs 56 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling on --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --reference-solve lagged --cut-transport sector --light-max-pieces 8192 --slices-c128 '' --slices-light-c128 '' --ell 8 \
 --slice-dir $O/states --out $O --tag C64_s8_ell8_tau2_twofield_v4L --resume-slice $S --resume-run $H/snap/C64_s8_ell8_tau2_twofield_v4L --segment-end-T-over-rs 13.1875 --anchor-passage lz $ON $BASE $X > $O/engine_run.log 2>&1
rc=$?
echo "end $(date -u +%FT%TZ) rc $rc wall $(( $(date +%s) - t0 )) s"
