#!/bin/bash
# pod5_queue_v8.sh (solve builder 8, analyst; engine spec 49.1 (2), 49.2 (1), 49.0 (ii)): replaces pod5_queue_v7.sh (stopped by its PID), keeping
# v7's monitor lines (every 10 min: free memory, the shares, REPORT below 8192 MiB). Jobs, one at a time, beside the pair's member, read in
# order from ${QI2_ROOT}/pod5_jobs_v8.txt (lines "tag|need_MiB|command", appended by the builder; done tags in ${QI2_ROOT}/pod5_jobs_v8.done):
# a job starts only when the card has need_MiB + 8192 MiB free, and not in the gap between the ell 2a member's end and the ell 4a member's
# start (a running job finishes). The guard: below 8192 MiB free the placed job (its PID, command line checked) is stopped and re-queued (its
# own resume where it has one); the members are never stopped. A job exiting nonzero twice is left for the builder (REPORT).
# Usage (pod 5): setsid nohup bash pod5_queue_v8.sh > /dev/null 2>&1 < /dev/null &
set -u
W=/workspace; LOG=$W/pod5_queue_v8.log; JOBS=$W/pod5_jobs_v8.txt; DONE=$W/pod5_jobs_v8.done
touch $JOBS $DONE
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
shares() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';'; }
m2a() { ps -eo pid,args | grep "[e]ngine_collapse_v36.py.*pair_rs32_ell2a_v36" | grep -v bash | awk '{print $1}' | head -1; }
m4a() { ps -eo pid,args | grep "[e]ngine_collapse_v36b.py.*pair_rs32_ell4a_v36b" | grep -v bash | awk '{print $1}' | head -1; }
gap() { [ -z "$(m2a)" ] && [ -z "$(m4a)" ]; }   # the ell 2a member ended, the ell 4a member not yet running
LASTMON=0
monitor() { local now=$(date +%s); [ $((now - LASTMON)) -ge 600 ] || return 0; LASTMON=$now; local f=$(free_mib)
  log "free $f MiB; shares: $(shares); ell 2a PID $(m2a); ell 4a PID $(m4a)"; [ "${f:-99999}" -lt 8192 ] && log "REPORT: free $f MiB below 8192"; return 0; }
log "queue v8 start pid $$"
while true; do
  monitor
  line=$(while IFS= read -r l; do t=${l%%|*}; [ -n "$t" ] && ! grep -qx "$t" $DONE && { echo "$l"; break; }; done < $JOBS)
  if [ -z "$line" ]; then sleep 60; continue; fi
  tag=${line%%|*}; rest=${line#*|}; need=${rest%%|*}; cmd=${rest#*|}
  f=$(free_mib)
  if gap || [ "${f:-0}" -lt $((need + 8192)) ]; then sleep 60; continue; fi
  tries=0
  while true; do
    log "start $tag (need $need MiB; free $(free_mib) MiB; shares $(shares))"
    setsid bash -c "$cmd" >> $W/logs_v8_$tag.log 2>&1 &
    P=$!; G=0
    while [ -e /proc/$P ]; do
      sleep 30; monitor; f=$(free_mib)
      if [ "${f:-99999}" -lt 8192 ]; then
        log "GUARD: free $f MiB below 8192: $tag stopped (process group $P); shares $(shares)"; kill -- -$P 2>/dev/null; G=1
      fi
    done
    wait $P; rc=$?
    log "exit $tag $rc"
    if [ $G -eq 1 ]; then
      until ! gap && [ "$(free_mib)" -ge $((need + 8192)) ]; do sleep 60; monitor; done; log "$tag re-queued (guard)"; continue
    fi
    if [ $rc -eq 0 ]; then echo "$tag" >> $DONE; break; fi
    tries=$((tries + 1)); [ $tries -ge 2 ] && { log "REPORT: $tag exited nonzero twice: left for the builder"; echo "$tag" >> $DONE; break; }
    log "$tag resumed once"
  done
done
