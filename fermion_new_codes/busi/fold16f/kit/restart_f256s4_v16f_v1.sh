#!/bin/bash
# restart_f256s4_v16f_v1.sh (busi, analyst; ReviewA_s4fail_v1 sec. 2 and 3, inbox #444): F256_s4 restarts from T0000.06 on a rented pod under
# the field module v16f (= v16d 7a69faad + the fold exit; gated bitwise on the box), the engine 02a1729a, the line 052be409 with the Newton
# caches in host memory (lines_flat_taperW05_on_v3/F256_s4.args, md5 b72fd6c5: 052be409 with --nk-cache-host on; RD-1d's --nk-stage-release
# is not gated). ON THE POD, after flatgate_kit_v4's bootstrap (venv, peakhook) and with this kit unpacked anywhere.
#   RES=<dir holding the T0000.06 resume files, the box mirror rp-f256s4b/<tag>/T0000.06 copied whole> bash restart_f256s4_v16f_v1.sh
# It: installs ${QI2_ROOT}/code16f_flat from the kit (md5-checked; engine_fields_v16.py = v16f b5f7c12c); checks the line, engine,
# fields, banded md5s and the resume slice's md5 against RES/MD5SUMS; holds ${QI2_ROOT}/busi_gpu.lock; starts the engine with
# --resume-slice/--resume-run, the stall watcher (ReviewA_w256end_v1 sec. 3 (ii)), the slice keeper (the last two complete slices in
# states/, slice_keep_last2_v1.sh), and, once the run json exists, the stopper (birth_plus, --sigma 4) and the log-only check (logcheck 4),
# both reading the field module from code16f_flat. Slices: the line's own clocks (every 0.5 r_s; every 0.25 r_s from 3.0 to 6.0, w256end
# sec. 3 (i)). Never overwrites a run: refuses if $M/run/log.txt exists.
set -u
KIT=$(cd "$(dirname "$0")" && pwd); W=${QI2_ROOT}/flat; M=$W/F256_s4; CODE=${QI2_ROOT}/code16f_flat
PY=${PY:-${QI2_ROOT}/python}; [ -x $PY ] || PY=${QI2_ROOT}/python; THR=${THR:-14}; RES=${RES:?set RES}
TAG=lad_F256_s4_fbt05_m0p2_rs256_n4096_R04_k15_tau16_ell4_prep4_A_R2_avg_newton_v36nq_rd13mfb; A=$KIT/F256_s4.args
SL=$RES/orbitals_${TAG}_T0000.06.npz
[ "$(md5sum < $A | cut -c1-32)" = b72fd6c5732b6e8f125021ebbe5a3830 ] || { echo "line md5 (want b72fd6c5 = 052be409 with the cache on)"; exit 9; }
[ -e $CODE ] || { mkdir -p ${QI2_ROOT}/busi_flat && cp -a $KIT/code16f_flat $CODE; }
( cd $CODE && md5sum -c --quiet $KIT/MD5SUMS_code16f_flat.txt ) || { echo "code16f_flat md5"; exit 9; }
[ "$(md5sum < $CODE/engine_collapse_v36nq_rd13mfb.py | cut -c1-32)" = 02a1729ab808948d443eed9144b75b03 ] || { echo "engine md5"; exit 9; }
[ "$(md5sum < $CODE/engine_fields_v16.py | cut -c1-32)" = b5f7c12c259ed376fe96d5f14ca1fc49 ] || { echo "fields md5 (want v16f b5f7c12c)"; exit 9; }
[ "$(md5sum < $CODE/engine_banded_eigh_v1.py | cut -c1-32)" = 73ee46facc09b8cb77f095b2803a5c8e ] || { echo "banded md5"; exit 9; }
[ -e $SL ] && [ -e $RES/collapse_$TAG.json ] && [ -e $RES/collapse_${TAG}_fields.npz ] || { echo "RES lacks the slice, json or fields npz"; exit 9; }
want=$(awk -v f="$(basename $SL)" '{n=$2; sub(".*/","",n)} n==f{print $1}' $RES/MD5SUMS | head -1)
[ -n "$want" ] && [ "$(md5sum < $SL | cut -c1-32)" = "$want" ] || { echo "resume slice md5 against RES/MD5SUMS"; exit 9; }
[ -e $M/run/log.txt ] && { echo "$M/run/log.txt exists: move the earlier run aside first (never overwritten)"; exit 9; }
mkdir -p $M/run $M/states $M/logs
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $M/CHAIN.txt; }
exec 9>>${QI2_ROOT}/busi_gpu.lock
flock 9; log "lock held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "GPU busy, waiting"; sleep 30; done
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 5000 >> $M/logs/smi.csv 2>&1 &
SMI=$!; log "nvidia-smi logger pid $SMI"
ARGS1="$(cat $A) --resume-slice $SL --resume-run $RES/$TAG"
echo "$ARGS1" > $M/run/args.txt
md5sum $CODE/engine_collapse_v36nq_rd13mfb.py $CODE/engine_fields_v16.py $CODE/engine_banded_eigh_v1.py $A $SL $RES/collapse_$TAG.json > $M/run/RUN_MD5.txt
export OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=$THR NUMEXPR_NUM_THREADS=$THR NUMBA_NUM_THREADS=$THR NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE
export PYTHONPATH=${QI2_ROOT}/peakhook
log "run start (v16f restart from T0000.06, ReviewA_s4fail_v1): $PY -B -u $CODE/engine_collapse_v36nq_rd13mfb.py $ARGS1"
( cd $CODE && exec $PY -B -u $CODE/engine_collapse_v36nq_rd13mfb.py $ARGS1 ) > $M/run/log.txt 2>&1 < /dev/null &
P=$!; echo $P > $M/run/pid; log "run pid $P"
$PY $KIT/flat_stall_watch_v1.py $M/run/log.txt $P $M/logs/stallwatch.tsv --from-T 0.0625 > $M/logs/stallwatch.out 2>&1 &
log "stall watcher pid $! (ReviewA_w256end_v1 sec. 3 (ii))"
bash $KIT/slice_keep_last2_v1.sh $M $TAG $P > $M/logs/slice_keep.out 2>&1 &
log "slice keeper pid $! (the last two complete slices; ReviewA_s4fail_v1 sec. 3)"
( until [ -e $M/run/collapse_$TAG.json ] || ! kill -0 $P 2>/dev/null; do sleep 60; done
  if [ -e $M/run/collapse_$TAG.json ]; then
    CODE=$CODE PY=$PY bash $KIT/stopper_kit_v1/install_stopper_v2.sh 4 && CODE=$CODE PY=$PY bash $KIT/logcheck_kit_v1/install_logcheck_v2.sh 4
    log "stopper (birth_plus --sigma 4 --after-birth 1.25) and logcheck 4 installed on code16f_flat: rc $?"
  fi ) >> $M/logs/install_watchers.out 2>&1 &
wait $P; rc=$?; log "run rc $rc"; kill $SMI; flock -u 9; log "lock released"
