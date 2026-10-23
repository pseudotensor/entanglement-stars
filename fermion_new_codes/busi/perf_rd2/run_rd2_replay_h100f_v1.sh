#!/bin/bash
# run_rd2_replay_h100f_v1.sh (busi, analyst; PERF RD-2/RD-5, the ruling's sec. 4 gate): run_rd2_replay_v1.sh's replay extended to 8 records on
# h100-f (coordinator, inbox #383): R1 RUNG2c40b_srf from the mirrored T0012.00 slice to T/r_s 13.0 (12.125 ... 13.0), engine_collapse_v37a_rd2
# (ab68eb7d) in ${QI2_ROOT}/code16d (engine_fields_v16.py = v16d 7a69faad), the production's arguments from ARGV.txt with --gpu 0 and
# the paths moved, an end slice written at 13.0 (complex128). Arms one after the other on the same GPU, OMP/MKL/OPENBLAS 8, venv np1263:
#   I = --fields ivp (the default), C = --fields collocation, W = --fields collocation --fields-warm-start on.
# Each arm starts only when nvidia-smi shows no compute process (other builders share the card); it waits otherwise.
# usage: run_rd2_replay_h100f_v1.sh [ARMS]   (default "I C W"); outputs ${QI2_ROOT}/replay<arm>/
set -u
W=${QI2_ROOT}/busi_rd2; PY=${QI2_ROOT}/python; CODE=$W/code16d; MIR=$W/input/T0012.00
ARMS=${1:-"I C W"}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM H2_PW_ONE H2_LOCAL_SPLINE PYTHONPATH
ARGS="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --prep-ell 4 --ell 2 --tau 8 --T-over-rs 40 --gpu 0 --mix newton --nk-krylov 40 --nk-restarts 2 --nk-maxiter 30 --nk-halvings 6 --nk-inner 15 --nk-jvp exact --nk-line-search halving --nk-set resolve --end-above 0.001 --nk-host-release on --nk-plain-fallback on --step-refine on --step-refine-max-halvings 3 --field-kernel-cache --v36-fixes --resume-slice $MIR/orbitals_RUNG2c40b_srf_T0012.00.npz --resume-run $MIR/RUNG2c40b_srf --segment-end-T-over-rs 13.0 --slices-c128 13.0"
for arm in $ARMS; do
  case $arm in I) OPT="";; C) OPT="--fields collocation";; W) OPT="--fields collocation --fields-warm-start on";; *) echo "unknown arm $arm"; exit 2;; esac
  O=$W/replay/$arm
  [ -e $O/collapse_RUNG2c40b_srf_rd2$arm.json ] && { echo "$O exists; skipped"; continue; }
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "$(date -u +%FT%TZ) arm $arm waits: a compute process on the GPU" >> $W/replay_wait.log; sleep 60; done
  mkdir -p $O/states
  echo "$(date -u +%FT%TZ) arm $arm start: $CODE/engine_collapse_v37a_rd2.py $ARGS $OPT" > $O/RUN.txt
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv >> $O/RUN.txt
  ( cd $CODE && $PY -B engine_collapse_v37a_rd2.py $ARGS $OPT --out $O --tag RUNG2c40b_srf_rd2$arm --slice-dir $O/states ) > $O/replay_$arm.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) arm $arm rc $rc" >> $O/RUN.txt
  echo "arm $arm rc $rc $(date -u +%FT%TZ)" >> $W/replay_done.txt
done
