#!/bin/bash
# run_rd2_replay_v1.sh (busi, analyst; PERF RD-2/RD-5 self-test (c)): the engine-level replay of a stored Newton segment, R1 RUNG2c40b_srf
# from its mirrored T0012.00 slice to T/r_s 12.375 (3 records: 12.125, 12.25, 12.375), on engine_collapse_v37a_rd2 (code dir RD2_build/code16d,
# engine_fields_v16.py = v16d), the production's arguments from the mirror's ARGV.txt (v36q_srf on R1's H100) with the paths moved, no --gpu
# (box CPU), no slice written (--slices-c128 40.0 lies beyond the segment). Arms one after the other at 8 threads, nice 10:
#   I = --fields ivp (the default), C = --fields collocation, W = --fields collocation --fields-warm-start on.
# Outputs RD2_build/replay/<arm>/ (json, npz, log); the slice and the mirror are only read.   usage: run_rd2_replay_v1.sh [ARMS]   (default "I C W")
set -u
B=${QI2_ROOT}/RD2_build; PY=${QI2_ROOT}/python; CODE=$B/code16d
MIR=${QI2_ROOT}/T0012.00
ARMS=${1:-"I C W"}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM H2_PW_ONE H2_LOCAL_SPLINE PYTHONPATH
ARGS="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --prep-ell 4 --ell 2 --tau 8 --T-over-rs 40 --mix newton --nk-krylov 40 --nk-restarts 2 --nk-maxiter 30 --nk-halvings 6 --nk-inner 15 --nk-jvp exact --nk-line-search halving --nk-set resolve --end-above 0.001 --nk-host-release on --nk-plain-fallback on --step-refine on --step-refine-max-halvings 3 --field-kernel-cache --v36-fixes --resume-slice $MIR/orbitals_RUNG2c40b_srf_T0012.00.npz --resume-run $MIR/RUNG2c40b_srf --segment-end-T-over-rs 12.375 --slices-c128 40.0"
for arm in $ARMS; do
  case $arm in I) OPT="";; C) OPT="--fields collocation";; W) OPT="--fields collocation --fields-warm-start on";; *) echo "unknown arm $arm"; exit 2;; esac
  O=$B/replay/$arm
  [ -e $O/collapse_RUNG2c40b_srf_rd2$arm.json ] && { echo "$O exists; skipped"; continue; }
  mkdir -p $O/states
  echo "$(date -u +%FT%TZ) arm $arm start: $CODE/engine_collapse_v37a_rd2.py $ARGS $OPT" > $O/RUN.txt
  ( cd $CODE && nice -n 10 $PY -B engine_collapse_v37a_rd2.py $ARGS $OPT --out $O --tag RUNG2c40b_srf_rd2$arm --slice-dir $O/states ) > $O/replay_$arm.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) arm $arm rc $rc" >> $O/RUN.txt
  echo "arm $arm rc $rc $(date -u +%FT%TZ)" >> $B/logs/replay_done.txt
done
