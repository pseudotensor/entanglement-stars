#!/bin/bash
# restart_T3_h100f_v1.sh (busi, GATECOMP, the coordinator's slice-cadence order; ReviewA_flatq_v1 sec. 2): W256_s4 gets a slice every 1 r_s from 4 to 10 and every 0.25 r_s from 8.5 to 10.5 (the
# plan's list had 0, 3, 6, 9, then quarters from 10; a memory failure on a heavy record must leave a recent resumable slice). The slice list
# changes only at a restart: the first restart point is the member's own T0003.00 slice. ON THE CARD:
#  1. wait until T0003.00 exists, its size is stable for 60 s, and the json holds the 3.125 record (the json/npz are rewritten every 5
#     records, at 3.125 the last before 3.4375, so the npz then holds the 3.0 row and nothing is being written during 3.1875);
#  2. stop the engine by its PID (TERM, KILL after 120 s; never pkill), record it; the chain script then releases the lock;
#  3. the run's files are copied to run_preT3/ (md5 logged) and the member restarts from T0003.00 and that copy, with the same args and
#     the new slice list, into the SAME out dir run/ (so the resume audit's line stays valid; the json there carries the whole history
#     from T 0; the records 3.0625 to the stop are recomputed on the same route); the lock is taken, nvidia-smi logged.
set -u
W=${QI2_ROOT}/busi_gatecomp; M=$W/W256_s4; CODE=${QI2_ROOT}/code16d; PY=${QI2_ROOT}/python
TAG=lad_W256_s4_m0p2_rs256_n4096_R04_k14_tau16_ell4_prep4_A_R2_avg_newton_v36nq_rd13mf
SL=$M/states/orbitals_${TAG}_T0003.00.npz
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $M/RESTART_T3.txt; }
EP=${EP:?set EP to the engine pid}
log "waiting for $SL and the json's 3.125 record; engine pid $EP"
until [ -e $SL ] && s1=$(stat -c %s $SL) && sleep 60 && [ "$s1" = "$(stat -c %s $SL)" ] && \
      $PY -c "import json,sys; d=json.load(open('$M/run/collapse_$TAG.json')); sys.exit(0 if any(abs(r['T_over_rs']-3.125)<1e-9 for r in d['records']) else 1)" 2>/dev/null; do sleep 30; done
md5=$(md5sum < $SL | cut -c1-32); log "T0003.00 complete, md5 $md5; json holds 3.125; stopping engine pid $EP"
KIDS=$(ps -o pid= --ppid $EP | tr -s ' \n' ' '); log "engine children (pair workers): $KIDS"
kill -TERM $EP; for i in $(seq 1 24); do kill -0 $EP 2>/dev/null || break; sleep 5; done; kill -0 $EP 2>/dev/null && { kill -KILL $EP; log "KILL sent"; }
for k in $KIDS; do kill -0 $k 2>/dev/null && { kill -TERM $k; log "child $k TERM"; }; done
log "engine stopped; last records in the log: $(grep -E '^T/rs [0-9.]+ E/E0' $M/run/log.txt | tail -1 | cut -c1-40)"
exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9; log "lock held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 10; done
SLN="3.0000,4.0000,5.0000,6.0000,7.0000,8.0000,8.5000,8.7500,9.0000,9.2500,9.5000,9.7500,10.0000,10.2500,10.5000,10.7500,11.0000,11.2500,11.5000,11.7500,12.0000,12.5000"   # ReviewA_flatq_v1 sec. 2: every 1 r_s from 3, every 0.25 from 8.5 to 10.5 (its comparator root near 9.1); the plan's quarters to 12, 12.5 (the cap)
mkdir -p $M/run_preT3; cp -p $M/run/collapse_${TAG}* $M/run/log.txt $M/run/args.txt $M/run_preT3/; ( cd $M/run_preT3 && md5sum * > MD5SUMS ); log "run files copied to run_preT3 ($(md5sum < $M/run_preT3/MD5SUMS | cut -c1-8))"
mv $M/run/log.txt $M/run_preT3/log_full.txt 2>/dev/null
ARGS="$(sed -e "s#--slices-c128 [^ ]*#--slices-c128 $SLN#" $W/lines_h100f/W256_s4.args) --resume-slice $SL --resume-run $M/run_preT3/$TAG"
echo "$ARGS" > $M/run/args_T3.txt; md5sum $CODE/engine_collapse_v36nq_rd13mf.py $SL $M/run_preT3/collapse_$TAG.json > $M/run/RUN_MD5_T3.txt
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 NUMEXPR_NUM_THREADS=14 NUMBA_NUM_THREADS=14 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE
export PYTHONPATH=${QI2_ROOT}/peakhook
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 5000 >> $M/logs/smi.csv 2>&1 & SMI=$!
( cd $CODE && $PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $ARGS ) > $M/run/log.txt 2>&1 < /dev/null &
P2=$!; echo $P2 > $M/run/pid_T3; log "restart from T0003.00 ($md5) into run/, pid $P2, slices $SLN"
[ -n "${MW:-}" ] && kill $MW 2>/dev/null && log "memwatch $MW stopped (its log moved to run_preT3)"
setsid nohup $PY -B $W/code/memwatch_v1.py $M/run $TAG $M/run/log.txt $M/logs/smi.csv $M/logs/memwatch_T3.csv > /dev/null 2>&1 < /dev/null &
log "memwatch on the restarted run pid $!"
wait $P2; log "run (from T3) rc $?"; kill $SMI; flock -u 9; log "lock released"
