#!/bin/bash
# restart_w256_cont_h100f_v1.sh (busi, GATECOMP; ReviewA_w256end_v1 secs. 1 (alpha) and 2): W256_s4 restarted from its seg0 T0000.06 slice
# on h100-f, BEHIND the flat members (start it only when the flat member on the card has ended: it takes the shared lock and refuses a busy
# GPU), as the member, with --source-continuation two-stage (stage 2 = the polish, judged by --end-above 1e-3, unmoved), the stall watcher,
# K 14, banded, tau 16 a as staged (lines_h100f/W256_s4.args, md5 checked), T cap 12.5 (reviewer A sec. 2; the engine's --T-over-rs 14 replaced).
# Slices (sec. 2): every 0.5 r_s from 0.5, ROLLING (the newest two kept, every 2 r_s kept), and every 0.25 r_s from 8.0 to 10.5 (kept);
# the rolling pruner runs beside it. Out: W256_s4/run_c, states_c (the ended run's run/ and states/ untouched). ON THE CARD.
set -u
W=${QI2_ROOT}/busi_gatecomp; M=$W/W256_s4; CODE=${QI2_ROOT}/code16d; ENG=engine_collapse_v36nq_rd13mf.py; PY=${QI2_ROOT}/python
A=$W/lines_h100f/W256_s4.args; TAG=$(tr ' ' '\n' < $A | awk 'p{print; exit} $0=="--tag"{p=1}')
want=$(awk '$1=="W256_s4"{print $3}' $W/lines_h100f/LINES.tsv); [ "$(md5sum < $A | cut -c1-32)" = "$want" ] || { echo "args md5"; exit 9; }
[ "$(md5sum < $CODE/$ENG | cut -c1-32)" = 248f25f35e48a88a84b102ddd99a5713 ] || { echo "engine md5"; exit 9; }
SL=$M/seg0/states/orbitals_${TAG}_T0000.06.npz; [ -e $SL ] || { echo "no $SL"; exit 9; }
[ -e $M/run_c ] && { echo "$M/run_c exists"; exit 9; }
mkdir -p $M/run_c $M/states_c $M/logs
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $M/RESTART_W256_CONT.txt; }
SLN=$( (for x in $(seq 0.5 0.5 12.5); do echo $x; done; for x in $(seq 8.0 0.25 10.5); do echo $x; done) | sort -g -u | awk '{printf "%s%.4f", (NR>1?",":""), $1}')
ARGS1="$(sed -e "s#--slices-c128 [^ ]*#--slices-c128 $SLN#" -e "s#--slice-dir [^ ]*#--slice-dir $M/states_c#" -e "s#--out [^ ]*#--out $M/run_c#" -e "s#--T-over-rs 14#--T-over-rs 12.5#" $A) --source-continuation two-stage --resume-slice $SL --resume-run $M/seg0/run/$TAG"
echo "$ARGS1" > $M/run_c/args.txt
exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9; log "lock held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "GPU busy, waiting"; sleep 30; done
md5sum $CODE/$ENG $SL $M/seg0/run/collapse_$TAG.json > $M/run_c/RUN_MD5.txt
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 NUMEXPR_NUM_THREADS=14 NUMBA_NUM_THREADS=14 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE; export PYTHONPATH=${QI2_ROOT}/peakhook
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 5000 >> $M/logs/smi_c.csv 2>&1 & SMI=$!
( cd $CODE && exec $PY -B -u $CODE/$ENG $ARGS1 ) > $M/run_c/log.txt 2>&1 < /dev/null &
P=$!; echo $P > $M/run_c/pid; log "W256_s4 restart from T0000.06 (continuation two-stage, cap 12.5) engine pid $P"
$PY $W/flat/code/flat_stall_watch_v1.py $M/run_c/log.txt $P $M/logs/stallwatch_c.tsv --from-T 0.0625 > $M/logs/stallwatch_c.out 2>&1 & log "stall watcher pid $!"
( while kill -0 $P 2>/dev/null; do   # the rolling pruner: keep T0 links, the newest two of the 0.5 cadence, every 2 r_s, the dense 8.0 to 10.5
    ls $M/states_c/orbitals_${TAG}_T*.npz 2>/dev/null | sed -E 's/.*_T([0-9.]+)\.npz$/\1/' | sort -g > $M/logs/slices_c.txt
    n=$(wc -l < $M/logs/slices_c.txt)
    head -n $(( n > 2 ? n - 2 : 0 )) $M/logs/slices_c.txt | while read t; do
      awk -v t=$t 'BEGIN{x=t+0; keep=(x>=8.0-1e-9 && x<=10.5+1e-9) || (int(x/2+1e-9)*2==x)} END{exit !keep}' < /dev/null && continue
      f=$M/states_c/orbitals_${TAG}_T$t.npz; echo "$(date -u +%FT%TZ) pruned T$t $(md5sum < $f | cut -c1-32)" >> $M/logs/pruned_c.txt; rm -f $f
    done; sleep 600; done ) &
wait $P; log "run_c rc $?"; kill $SMI; flock -u 9; log "lock released"
