#!/bin/bash
# run_rd2_replay_h100f_v2.sh (busi, analyst; PERF RD-2/RD-5, REVIEWA_perf_tolerance_ruling_v1 sec. 4 on h100-f, inbox #383). v1's segment
# (RUNG2c40b_srf T0012.00 -> 13.0) cannot reach 8 records under the production's arguments: after 12.625 the srf step refinement fails and
# halves (12.75 failed at 8 a, 12.6875 at 4 a, 12.656 running at 2 a; the production R1 run itself stopped at 12.5 and its PTC continuation at
# 12.640625), so v1's arm I was stopped by its PIDs. Two segments instead, each arm one after the other on the same GPU, OMP/MKL/OPENBLAS 8:
#   S2 (the 8-record gate): R1 RUNG2c40_v36q from its mirrored T0002.00 slice to T/r_s 3.0 (2.125 ... 3.0, 8 records; the production's stored
#      records reach 3.125), engine engine_collapse_v36nq_rd2 (v36nq = v36q + options off; the production ran v36q), the production's ARGV word
#      for word with the paths moved, end slice at 3.0.
#   S1 (the set-change event, inversions 0 -> 4 at 12.375): RUNG2c40b_srf T0012.00 -> 12.5 (4 records), engine engine_collapse_v37a_rd2, the
#      production's ARGV, end slice at 12.5.
# Arms I (--fields ivp, the default), C (--fields collocation), W (C + --fields-warm-start on). Each arm starts only when nvidia-smi shows no
# compute process.   usage: run_rd2_replay_h100f_v2.sh [JOBS]   (default "S2:I S2:C S2:W S1:I S1:C S1:W"); outputs ${QI2_ROOT}/v2<seg>/<arm>/
set -u
W=${QI2_ROOT}/busi_rd2; PY=${QI2_ROOT}/python; CODE=$W/code16d
JOBS=${1:-"S2:I S2:C S2:W S1:I S1:C S1:W"}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM H2_PW_ONE H2_LOCAL_SPLINE PYTHONPATH
COMMON="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --prep-ell 4 --ell 2 --tau 8 --T-over-rs 40 --gpu 0 --mix newton --nk-krylov 40 --nk-restarts 2 --nk-maxiter 30 --nk-halvings 6 --nk-inner 15 --nk-jvp exact --nk-line-search halving --nk-set resolve --end-above 0.001 --nk-host-release on --nk-plain-fallback on"
M2=$W/input_v36q/T0002.00; M1=$W/input/T0012.00
for job in $JOBS; do
  seg=${job%%:*}; arm=${job##*:}
  case $seg in
    S2) ENG=engine_collapse_v36nq_rd2.py; TAG=RUNG2c40_v36q_rd2$arm
        ARGS="$COMMON --field-kernel-cache --v36-fixes --resume-slice $M2/orbitals_RUNG2c40_v36q_T0002.00.npz --resume-run $M2/RUNG2c40_v36q --segment-end-T-over-rs 3.0 --slices-c128 3.0";;
    S1) ENG=engine_collapse_v37a_rd2.py; TAG=RUNG2c40b_srf_rd2$arm
        ARGS="$COMMON --step-refine on --step-refine-max-halvings 3 --field-kernel-cache --v36-fixes --resume-slice $M1/orbitals_RUNG2c40b_srf_T0012.00.npz --resume-run $M1/RUNG2c40b_srf --segment-end-T-over-rs 12.5 --slices-c128 12.5";;
    *) echo "unknown segment $seg"; exit 2;;
  esac
  case $arm in I) OPT="";; C) OPT="--fields collocation";; W) OPT="--fields collocation --fields-warm-start on";; *) echo "unknown arm $arm"; exit 2;; esac
  O=$W/v2/$seg/$arm
  [ -e $O/collapse_$TAG.json ] && { echo "$O exists; skipped"; continue; }
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "$(date -u +%FT%TZ) $job waits: a compute process on the GPU" >> $W/v2/wait.log; sleep 60; done
  mkdir -p $O/states
  echo "$(date -u +%FT%TZ) $job start: $CODE/$ENG $ARGS $OPT" > $O/RUN.txt
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv >> $O/RUN.txt
  ( cd $CODE && $PY -B $ENG $ARGS $OPT --out $O --tag $TAG --slice-dir $O/states ) > $O/replay.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $job rc $rc" >> $O/RUN.txt
  echo "$job rc $rc $(date -u +%FT%TZ)" >> $W/v2/done.txt
done
