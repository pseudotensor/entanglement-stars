#!/bin/bash
# pod1_queue_v1.sh (solve builder 8, analyst; engine spec 50.3): pod5_queue_v8.sh's logic for pod 1 beside A2 (A2 never stopped; its own
# guard_pod1_A2_v1 untouched), without pod 5's member gap. Monitor lines every 10 min (free memory, the shares, REPORT below 8192 MiB). Jobs, one
# at a time, read in order from ${QI2_ROOT}/pod1_jobs_v1.txt (lines "tag|need_MiB|command"; done tags in ${QI2_ROOT}/pod1_jobs_v1.done):
# a job starts only when the card has need_MiB + 8192 MiB free. The guard: below 8192 MiB free the placed job (its PID, command line checked) is stopped and re-queued (its
# own resume where it has one); the members are never stopped. A job exiting nonzero twice is left for the builder (REPORT).
# Usage (pod 1): setsid nohup bash pod1_queue_v1.sh > /dev/null 2>&1 < /dev/null &
set -u
W=/workspace; LOG=$W/pod1_queue_v1.log; JOBS=$W/pod1_jobs_v1.txt; DONE=$W/pod1_jobs_v1.done
touch $JOBS $DONE
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
shares() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';'; }
m2a() { ps -eo pid,args | grep "[e]ngine_collapse_v36.py.*pair_rs32_ell2a_v36" | grep -v bash | awk '{print $1}' | head -1; }
m4a() { ps -eo pid,args | grep "[e]ngine_collapse_v36b.py.*pair_rs32_ell4a_v36b" | grep -v bash | awk '{print $1}' | head -1; }
gap() { return 1; }   # no member gap on pod 1
LASTMON=0
monitor() { local now=$(date +%s); [ $((now - LASTMON)) -ge 600 ] || return 0; LASTMON=$now; local f=$(free_mib)
  log "free $f MiB; shares: $(shares)"; [ "${f:-99999}" -lt 8192 ] && log "REPORT: free $f MiB below 8192"; return 0; }
log "pod 1 queue v1 start pid $$"
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
    setsid bash -c "$cmd" >> $W/logs_q1_$tag.log 2>&1 &
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
