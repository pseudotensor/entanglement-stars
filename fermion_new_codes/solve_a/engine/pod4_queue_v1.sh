#!/bin/bash
# pod4_queue_v1.sh (solve builder 10; pod3_queue_v1.sh for pod 4 under ${QI2_ROOT}/solve_a, starting only after B2 (its python, the B2r50 restart) has ended; the lead 23:40 UTC: pod 4 takes the graft once B2 ends). Version of: pod3_queue_v1.sh (solve builder 10, analyst; engine spec 52.2 (1), (3)): the solve lane's jobs on the busi lane's offered pod 3 H100,
# one at a time, under ${QI2_ROOT}/solve_a only (never ${QI2_ROOT}/busi_compC, never the busi lane's processes or diagnostics).
# Jobs, in order, from ${QI2_ROOT}/pod3_jobs_v1.txt (lines "tag|need_MiB|command", run from ${QI2_ROOT}/solve_a; done tags
# in pod3_jobs_v1.done). A job starts only when the card has need_MiB + 8192 MiB free. Its device use is sampled every 30 s by its PID
# (nvidia-smi --query-compute-apps) and the peak printed at its end ("peak <MiB>"). The guard: below 8192 MiB free the job (this queue's own
# child, its process group) is stopped and re-queued. The busi lane's word: the file ${QI2_ROOT}/YIELD present stops the running
# job the same way and holds the queue until the file is removed (the job then re-queued whole). Log: ${QI2_ROOT}/pod3_queue_v1.log.
W=${QI2_ROOT}/solve_a; LOG=$W/pod4_queue_v1.log; JOBS=$W/pod4_jobs_v1.txt; DONE=$W/pod4_jobs_v1.done
cd $W || exit 1; touch $DONE
log() { echo "$(date -u '+%F %T UTC') $*" >> $LOG; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
used_by() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, -v p=" $1" -v q="$1" '$1==q||$1==p{s+=$2} END{print s+0}'; }
pids_in_group() { ps -eo pid,pgid | awk -v g=$1 '$2==g{print $1}'; }
log "pod 4 queue v1 start pid $$; waiting for B2 to end"; while ps -eo args | grep -q "[r]un_collapse_v19_from_slice_v1.py.*B2"; do sleep 60; done; log "B2 ended"
while true; do
  if [ -e $W/YIELD ]; then sleep 60; continue; fi
  line=$(while IFS= read -r l; do t=${l%%|*}; [ -n "$t" ] && ! grep -qx "$t" $DONE && { echo "$l"; break; }; done < $JOBS)
  if [ -z "$line" ]; then sleep 60; continue; fi
  tag=${line%%|*}; rest=${line#*|}; need=${rest%%|*}; cmd=${rest#*|}
  f=$(free_mib); if [ "${f:-0}" -lt $((need + 8192)) ]; then sleep 60; continue; fi
  tries=0
  while true; do
    log "start $tag (need $need MiB; free $(free_mib) MiB)"
    setsid bash -c "$cmd" >> $W/logs/q3_$tag.log 2>&1 &
    P=$!; G=0; peak=0
    while [ -e /proc/$P ]; do
      sleep 30; f=$(free_mib); u=0
      for q in $(pids_in_group $P); do u=$((u + $(used_by $q))); done
      [ $u -gt $peak ] && peak=$u
      if [ "${f:-99999}" -lt 8192 ]; then log "GUARD: free $f MiB below 8192: $tag stopped (process group $P)"; kill -- -$P 2>/dev/null; G=1; fi
      if [ -e $W/YIELD ] && [ $G -eq 0 ]; then log "YIELD: $tag stopped at the busi lane's word (process group $P)"; kill -- -$P 2>/dev/null; G=1; fi
    done
    wait $P; rc=$?
    log "exit $tag $rc; peak $peak MiB"
    if [ $G -eq 1 ]; then
      until [ ! -e $W/YIELD ] && [ "$(free_mib)" -ge $((need + 8192)) ]; do sleep 60; done; log "$tag re-queued"; continue
    fi
    if [ $rc -eq 0 ]; then echo "$tag" >> $DONE; break; fi
    tries=$((tries + 1)); [ $tries -ge 2 ] && { log "REPORT: $tag exited nonzero twice: left for the builder"; echo "$tag" >> $DONE; break; }
    log "$tag resumed once"
  done
done
