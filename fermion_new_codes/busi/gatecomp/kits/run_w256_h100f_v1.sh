#!/bin/bash
# run_w256_h100f_v1.sh (busi, GATECOMP kit (ii), #400/#402; ReviewA_s3b_v1 sec. 3): one W256 member on h100-f, entered as reviewer A's sec. 3
# says. ON THE CARD, from ${QI2_ROOT}/busi_gatecomp. usage: run_w256_h100f_v1.sh S   (S = 4, 8 or 16)
#  seg0: the member's own line (lines_h100f/W256_sS.args, md5 checked against LINES.tsv) to --segment-end-T-over-rs 0.0625 with slices
#        0 and 0.0625 (the preparation and the first two records; the T = 0 slice is the dust comparator's profile);
#  percall: rd13_percall_v1.py (busi/code/rd13, RD-1's per-call test) on the T = 0 and T = 0.0625 slices' own matrices, channels 1, 7, 14:
#        banded against dense eigenvalues, projectors, tracked sets, J.v;
#  GO:   waits for W256_sS/GO (written by busi after reading the per-call test) or W256_sS/NOGO (exit);
#  run:  the member from seg0's 0.0625 slice (a version-34 restart with the same args) to T 14, the line's slice list.
# The shared lock ${QI2_ROOT}/busi_gpu.lock is held for the whole chain; nvidia-smi is logged every 5 s; the peak hook prints torch's
# max_memory_allocated/reserved at each process's exit. Threads 14 of 16 (H1).
set -u
S=$1; W=${QI2_ROOT}/busi_gatecomp; CODE=${QI2_ROOT}/code16d; PY=${QI2_ROOT}/python; M=$W/W256_s$S
A=$W/lines_h100f/W256_s$S.args; TAG=$(tr ' ' '\n' < $A | awk 'p{print; exit} $0=="--tag"{p=1}')
want=$(awk -v n=W256_s$S '$1==n{print $3}' $W/lines_h100f/LINES.tsv); [ "$(md5sum < $A | cut -c1-32)" = "$want" ] || { echo "args md5"; exit 9; }
[ "$(md5sum < $CODE/engine_collapse_v36nq_rd13mf.py | cut -c1-32)" = 248f25f35e48a88a84b102ddd99a5713 ] || { echo "engine md5"; exit 9; }
[ "$(md5sum < $CODE/engine_fields_v16.py | cut -c1-32)" = 7a69faadfad4a1e824a41107f474c69c ] || { echo "fields md5"; exit 9; }
[ "$(md5sum < $CODE/engine_banded_eigh_v1.py | cut -c1-32)" = 73ee46facc09b8cb77f095b2803a5c8e ] || { echo "banded md5"; exit 9; }
mkdir -p $M/seg0/run $M/seg0/states $M/run $M/states $M/logs $M/percall
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 NUMEXPR_NUM_THREADS=14 NUMBA_NUM_THREADS=14 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE
export PYTHONPATH=${QI2_ROOT}/peakhook
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $M/CHAIN.txt; }
exec 9>>${QI2_ROOT}/busi_gpu.lock
flock 9; log "lock held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "GPU busy, waiting"; sleep 30; done
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 5000 >> $M/logs/smi.csv 2>&1 &
SMI=$!; log "nvidia-smi logger pid $SMI"
if [ ! -e $M/seg0/DONE ]; then
  ARGS0="$(sed -e "s#--slices-c128 [^ ]*#--slices-c128 0.0000,0.0625#" -e "s#--slice-dir [^ ]*#--slice-dir $M/seg0/states#" -e "s#--out [^ ]*#--out $M/seg0/run#" $A) --segment-end-T-over-rs 0.0625"
  echo "$ARGS0" > $M/seg0/args.txt
  log "seg0 start: $PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $ARGS0"
  ( cd $CODE && $PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $ARGS0 ) > $M/seg0/log.txt 2>&1 < /dev/null &
  P=$!; echo $P > $M/seg0/pid; log "seg0 pid $P"; wait $P; rc=$?; log "seg0 rc $rc"
  [ $rc = 0 ] || { kill $SMI; exit 1; }
  touch $M/seg0/DONE; md5sum $M/seg0/states/*.npz $M/seg0/run/* >> $M/seg0/MD5SUMS 2>/dev/null
fi
F0=$M/seg0/run/collapse_${TAG}_fields.npz
for t in 0000.00 0000.06; do
  [ -e $M/percall/T$t/rd13_percall_v1.json ] && continue
  mkdir -p $M/percall/T$t
  ( cd $CODE && $PY -B $W/code/rd13_percall_v1.py $M/percall/T$t $CODE --iso $M/seg0/states/orbitals_${TAG}_T$t.npz --iso-fields $F0 --channels-iso 1,7,14 ) > $M/percall/T$t/log.txt 2>&1
  log "percall T$t rc $?"
done
log "waiting for $M/GO or $M/NOGO"
while [ ! -e $M/GO ] && [ ! -e $M/NOGO ]; do sleep 20; done
[ -e $M/NOGO ] && { log "NOGO: stop"; kill $SMI; exit 0; }
SL1=$M/seg0/states/orbitals_${TAG}_T0000.06.npz
ARGS1="$(cat $A) --resume-slice $SL1 --resume-run $M/seg0/run/$TAG"
echo "$ARGS1" > $M/run/args.txt
log "run start (GO): $PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $ARGS1"
md5sum $CODE/engine_collapse_v36nq_rd13mf.py $CODE/engine_fields_v16.py $CODE/engine_banded_eigh_v1.py $SL1 $M/seg0/run/collapse_${TAG}*.json >> $M/run/RUN_MD5.txt
( cd $CODE && $PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $ARGS1 ) > $M/run/log.txt 2>&1 < /dev/null &
P=$!; echo $P > $M/run/pid; log "run pid $P"; wait $P; rc=$?; log "run rc $rc"
kill $SMI; flock -u 9; log "lock released"
