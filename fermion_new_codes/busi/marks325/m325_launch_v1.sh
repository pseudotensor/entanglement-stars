#!/bin/bash
# m325_launch_v1.sh (busi f52b1152, 2026-10-20; inbox #325, the far dust on the completions): one launcher, three modes, on one GPU each.
#   own56  the record's own history (rung 4, c32 v36d from T = 0 to 40) continued 40 -> 56 on its own engine v36d from its T0040 slice:
#          the matched charge target the held-source (frozen-sources) completion needs past 40, and the own column of the far-dust table.
#   pa56   the widened (paired-avg) completion continued 40 -> 56 from its own T0040 slice on v36g; the completion re-arms at the first
#          record (40.125, ungated at 40: gate radius 37.5 a) with the original R_* forced (--completion-radius-over-rs 0.3125 = 20 a)
#          and ell_in 8 a, the only state paired-avg carries.
#   fs56   the held-source (frozen-sources) completion RERUN from the 11.0 slice (its original resume slice) to 56 on v36g with the
#          original chain's arguments, the matched target read from own56's json (records 0 to 40 copied from the to40 run, 40.125 to
#          56 its own). It cannot resume at 40: v36g keeps the gated record's frozen sources only in memory, a run whose first solved
#          record is ungated stops ("frozen-sources needs the gated record's sources from a record this run solved"), and the
#          matched target has no record past 40 except own56's. Needs own56 finished (rc 0, last record 56): refused otherwise.
# Slices (complex128) at 44, 48, 50, 52, 54, 56 (fs56 also 40, the regression against the original run's T0040 slice md5 132ac218).
# Model: ${QI2_ROOT}/w2p_launch_busi.f4c_anc.sh (refusal if any output exists, nohup setsid, pid and rc files).
# Usage: m325_launch_v1.sh own56|pa56|fs56 [print|dry]   (print: the command and md5s; dry: also m325_dry_v1.py, argparse and the resume checks, no solve)
# Environment (defaults for a card): M325_KIT=${QI2_ROOT}/kit_v1  M325_OUT=${QI2_ROOT}/marks325
#   M325_PY=${QI2_ROOT}/python  M325_GPU=0
set -e
MODE=$1
K=${M325_KIT:-${QI2_ROOT}/kit_v1}; OUT=${M325_OUT:-${QI2_ROOT}/marks325}
PY=${M325_PY:-${QI2_ROOT}/python}; GPU=${M325_GPU:-0}
case "$MODE" in own56|pa56|fs56) ;; *) echo "usage: $0 own56|pa56|fs56 [print|dry]"; exit 1;; esac
BASE="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --prep-ell 4 --ell 4 --tau 8 --gpu $GPU --mix plain --field-kernel-cache --v36-fixes"
SL56=44.0000000000,48.0000000000,50.0000000000,52.0000000000,54.0000000000,56.0000000000
OWNTAG=c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40
OWNRUN=$K/inputs/own/runs/$OWNTAG
OWN56TAG=c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to56m325
PATAG=w3_paired-avg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36g
FSTAG=w3_frozen-sources_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36g
case "$MODE" in
  own56)
    ENGINE=$K/code/engine_collapse_v36d.py; TAG=$OWN56TAG
    SLICE=$K/inputs/own/states/orbitals_${OWNTAG}_T0040.00.npz; RRUN=$OWNRUN
    EXTRA="--T-over-rs 56 --slices-c128 $SL56";;
  pa56)
    ENGINE=$K/code/engine_collapse_v36g.py; TAG=${PATAG}_to56m325
    SLICE=$K/inputs/w3_paired-avg/states/orbitals_${PATAG}_T0040.00.npz; RRUN=$K/inputs/w3_paired-avg/runs/$PATAG
    EXTRA="--T-over-rs 56 --slices-c128 $SL56 --resume-args-check relaxed --completion paired-avg --completion-from-T-over-rs 40 --completion-radius-over-rs 0.3125 --completion-ell-in 8";;
  fs56)
    ENGINE=$K/code/engine_collapse_v36g.py; TAG=${FSTAG}_to56m325
    SLICE=$K/inputs/own/states/orbitals_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_T0011.00.npz; RRUN=$OWNRUN
    CRUN=$OUT/runs/own56/$OWN56TAG
    EXTRA="--T-over-rs 56 --slices-c128 40.0000000000,$SL56 --resume-args-check relaxed --completion frozen-sources --completion-from-T-over-rs 11.125 --completion-charge matched --completion-charge-tol 1e-9 --completion-run $CRUN";;
esac
RUN=$OUT/runs/$MODE; STD=$OUT/states/$MODE; LOG=$OUT/logs/$MODE.log; PIDF=$OUT/pids/$MODE.pid
ARGS="$BASE $EXTRA --slice-dir $STD --out $RUN --tag $TAG --resume-slice $SLICE --resume-run $RRUN"
echo "mode $MODE: $PY $ENGINE $ARGS" | fold -w 250
echo "engine $ENGINE md5 $(md5sum < $ENGINE | cut -c1-8); slice $(basename $SLICE) md5 $(md5sum < $SLICE | cut -c1-8)"
if [ "$MODE" = fs56 ]; then
  J=$OUT/runs/own56/collapse_$OWN56TAG.json
  if [ -e $J ]; then
    LAST=$($PY -c "import json,sys; J=json.load(open('$J')); print('%.4f' % (J['records'][-1]['T'] / 64.0))")
    RCO=$(cat $OUT/pids/own56.pid.rc 2>/dev/null || echo none)
    echo "own56 (the matched target): last record T/rs $LAST, rc $RCO"
    if [ "$2" != print ] && [ "$2" != dry ] && { [ "$LAST" != 56.0000 ] || [ "$RCO" != 0 ]; }; then echo "refusing: own56 has not finished to 56 with rc 0"; exit 1; fi
  else
    echo "own56's json is not there yet ($J)"; [ "$2" = print ] || [ "$2" = dry ] || { echo "refusing: fs56 needs own56 first"; exit 1; }
  fi
fi
if [ "$2" = dry ]; then
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10 $PY $K/code/m325_dry_v1.py $ENGINE $ARGS; exit $?
fi
if [ "$2" != print ]; then
  for f in $RUN $STD $LOG $PIDF $PIDF.rc; do test ! -e $f || { echo "refusing: $f exists"; exit 1; }; done
  mkdir -p $RUN $STD $OUT/logs $OUT/pids
  export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd $RUN
  setsid nohup bash -c "nice -n 10 $PY $ENGINE $ARGS > $LOG 2>&1 < /dev/null & echo \$! > $PIDF; wait \$!; echo \$? > $PIDF.rc" > /dev/null 2>&1 < /dev/null &
  sleep 3; echo "launched $MODE pid $(cat $PIDF) log $LOG"
fi
