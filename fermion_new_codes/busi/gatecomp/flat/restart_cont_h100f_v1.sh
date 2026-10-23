#!/bin/bash
# restart_cont_h100f_v1.sh (busi, GATECOMP; ReviewA_w256end_v1 sec. 3 (ii)): restart a running member from its latest slice at a record
# boundary with --source-continuation two-stage (the polish: stage 2 is the plain Newton on this record's map from the continuation's
# endpoint, judged by --end-above 1e-3), plus the stall watcher. ON THE CARD. The pattern of restart_T3_h100f_v1.sh:
#  1. wait for a slice of the member that the json covers (the json and npz are rewritten every 5 records): the newest slice T_s with a json
#     record at T >= T_s + 1/16, the slice's size stable for 60 s;
#  2. stop the engine by its PID (TERM, KILL after 120 s; its children; never pkill); the member's chain script then releases the lock;
#  3. copy the run files to run_pre<T_s>/ (md5 logged); restart from T_s and that copy, same args + the continuation (+ EXTRA), into the SAME
#     out dir run/ (the audit's line stays valid); the lock taken, nvidia-smi logged; the watcher started on the new engine pid.
# usage: EP=<engine pid> M=<member dir> CODE=<code dir> ENG=<engine file> ARGS=<the member's args file> [EXTRA="..."] restart_cont_h100f_v1.sh
set -u
PY=${PY:-${QI2_ROOT}/python}; EP=${EP:?}; M=${M:?}; CODE=${CODE:?}; ENG=${ENG:?}; A=${ARGS:?}; EXTRA=${EXTRA:-}
TAG=$(tr ' ' '\n' < $A | awk 'p{print; exit} $0=="--tag"{p=1}'); J=$M/run/collapse_$TAG.json
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $M/RESTART_CONT.txt; }
log "restart with the continuation: engine pid $EP, member $M, tag $TAG"
pick() { $PY - "$M/states" "$TAG" "$J" <<'PYX'
import glob, json, os, re, sys
d, tag, jf = sys.argv[1:4]
try:
    last = max(r["T_over_rs"] for r in json.load(open(jf))["records"])
except Exception:
    sys.exit(1)
ts = sorted(float(re.search(r"_T(\d+\.\d+)\.npz$", p).group(1)) for p in glob.glob(os.path.join(d, "orbitals_%s_T*.npz" % tag)))
ok = [t for t in ts if t + 0.0625 - 1e-9 <= last and t > 0.07]
print("%07.2f" % ok[-1] if ok else ""); sys.exit(0 if ok else 1)
PYX
}
until T=$(pick) && [ -n "$T" ] && SL=$M/states/orbitals_${TAG}_T$T.npz && s1=$(stat -L -c %s $SL) && sleep 60 && [ "$s1" = "$(stat -L -c %s $SL)" ]; do sleep 20; done
md5=$(md5sum < $SL | cut -c1-32); log "slice T$T complete (md5 $md5), the json covers it; stopping engine pid $EP"
KIDS=$(ps -o pid= --ppid $EP | tr -s ' \n' ' ')
kill -TERM $EP; for i in $(seq 1 24); do kill -0 $EP 2>/dev/null || break; sleep 5; done; kill -0 $EP 2>/dev/null && { kill -KILL $EP; log "KILL sent"; }
for k in $KIDS; do kill -0 $k 2>/dev/null && kill -TERM $k; done
log "engine stopped; last record in the log: $(grep -E '^T/rs [0-9.]+ E/E0' $M/run/log.txt | tail -1 | cut -c1-40)"
sleep 5
exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9; log "lock held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "GPU busy, waiting"; sleep 15; done
P0=$M/run_preT$T; mkdir -p $P0; cp -p $M/run/collapse_${TAG}* $M/run/log.txt $P0/; md5sum $P0/* > $P0/MD5SUMS; log "run files copied to $P0"
ARGS1="$(cat $A) --source-continuation two-stage $EXTRA --resume-slice $SL --resume-run $P0/$TAG"
echo "$ARGS1" > $M/run/args_cont_T$T.txt; md5sum $CODE/$ENG $SL $P0/collapse_$TAG.json > $M/run/RUN_MD5_cont_T$T.txt
export OMP_NUM_THREADS=${THR:-14} MKL_NUM_THREADS=${THR:-14} OPENBLAS_NUM_THREADS=${THR:-14} NUMEXPR_NUM_THREADS=${THR:-14} NUMBA_NUM_THREADS=${THR:-14} NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE; export PYTHONPATH=${QI2_ROOT}/peakhook
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 5000 >> $M/logs/smi.csv 2>&1 & SMI=$!
mv $M/run/log.txt $M/run/log_preT$T.txt
( cd $CODE && exec $PY -B -u $CODE/$ENG $ARGS1 ) > $M/run/log.txt 2>&1 < /dev/null &   # exec: $! is the engine itself
P2=$!; echo $P2 > $M/run/pid; log "run (from T$T, continuation two-stage) pid $P2"
$PY $(dirname $0)/flat_stall_watch_v1.py $M/run/log.txt $P2 $M/logs/stallwatch.tsv --from-T $T > $M/logs/stallwatch.out 2>&1 &
log "stall watcher pid $!"
wait $P2; log "run (from T$T) rc $?"; kill $SMI; flock -u 9; log "lock released"
