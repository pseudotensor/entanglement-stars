#!/bin/bash
# run_rd2b_replay_h100f_v1.sh (busi, analyst; PERF RD-2b self-test (c), REVIEWA_perf_tolerance_ruling_v1 sec. 4 on h100-f): RD-2's 8-record gate
# segment S2 again, now on engine_fields_v16e: R1 RUNG2c40_v36q from its mirrored T0002.00 slice to T/r_s 3.0 (2.125 ... 3.0; the records
# 2.125, 2.375, 2.625 and 2.875 have the negative centre that sent every RD-2 solve there to the ivp route), engine engine_collapse_v36nq_rd2
# (186a4ccd, unchanged) in ${QI2_ROOT}/code16e (RD-2's code16d with engine_fields_v16.py = v16e's content), the production's ARGV word
# for word with the paths moved (RD-2's run_rd2_replay_h100f_v2.sh, S2), end slice at 3.0. Arms I (--fields ivp, the default) and C (--fields
# collocation), one after the other on the same GPU, OMP/MKL/OPENBLAS 8; each starts only when nvidia-smi shows no compute process.
# usage: run_rd2b_replay_h100f_v1.sh [JOBS]   (default "S2:I S2:C"); outputs ${QI2_ROOT}/busi_rd2b<seg>/<arm>/
set -u
W=${QI2_ROOT}/busi_rd2b; PY=${QI2_ROOT}/python; CODE=$W/code16e
JOBS=${1:-"S2:I S2:C"}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE PYTHONPATH
COMMON="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --prep-ell 4 --ell 2 --tau 8 --T-over-rs 40 --gpu 0 --mix newton --nk-krylov 40 --nk-restarts 2 --nk-maxiter 30 --nk-halvings 6 --nk-inner 15 --nk-jvp exact --nk-line-search halving --nk-set resolve --end-above 0.001 --nk-host-release on --nk-plain-fallback on"
M2=${QI2_ROOT}/T0002.00
for job in $JOBS; do
  seg=${job%%:*}; arm=${job##*:}
  case $seg in
    S2) ENG=engine_collapse_v36nq_rd2.py; TAG=RUNG2c40_v36q_rd2b$arm
        ARGS="$COMMON --field-kernel-cache --v36-fixes --resume-slice $M2/orbitals_RUNG2c40_v36q_T0002.00.npz --resume-run $M2/RUNG2c40_v36q --segment-end-T-over-rs 3.0 --slices-c128 3.0";;
    *) echo "unknown segment $seg"; exit 2;;
  esac
  case $arm in I) OPT="";; C) OPT="--fields collocation";; *) echo "unknown arm $arm"; exit 2;; esac
  O=$W/$seg/$arm
  [ -e $O/collapse_$TAG.json ] && { echo "$O exists; skipped"; continue; }
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "$(date -u +%FT%TZ) $job waits: a compute process on the GPU" >> $W/wait.log; sleep 60; done
  mkdir -p $O/states
  echo "$(date -u +%FT%TZ) $job start: $CODE/$ENG $ARGS $OPT" > $O/RUN.txt
  md5sum $CODE/$ENG $CODE/engine_fields_v16.py >> $O/RUN.txt
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv >> $O/RUN.txt
  ( cd $CODE && $PY -B $ENG $ARGS $OPT --out $O --tag $TAG --slice-dir $O/states ) > $O/replay.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $job rc $rc" >> $O/RUN.txt
  echo "$job rc $rc $(date -u +%FT%TZ)" >> $W/done.txt
done
