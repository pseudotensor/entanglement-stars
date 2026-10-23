#!/bin/bash
# guard_pod1_A2_v1.sh (the lane's analyst builder; engine spec 46.2 (1)): the free-memory guard on pod 1's H100. Every 60 s the card's free
# memory is read; below 8192 MiB the placed process, A2 (page_history_v6i ... v6iA2_B2, the builder's own), is stopped by its PID (command line
# read first), its records and resumable state kept; it re-queues in this venue: relaunched by launch_A2_pod1_v1.sh (--resume, from its saved
# state) once 45000 MiB are free (its peak 35.8 GiB plus 8 GB). The tightened v36 run is never stopped. Every 10 min every process's device
# share is printed. Ends when A2 ends by itself (its exit line in logs/v6iA2_B2.status). Usage (pod 1): setsid nohup bash guard_pod1_A2_v1.sh &
L=${QI2_ROOT}/guard_pod1_A2_v1.log; ST=${QI2_ROOT}/v6iA2_B2.status
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $L; }
apid() { ps -eo pid,args | grep "[p]age_history_v6i.py.*v6iA2_B2" | awk '{print $1}' | head -1; }
log "guard v1 start; A2 PID $(apid)"
STOPPED=0; n=0
while true; do
  P=$(apid); FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ -z "$P" ]; then
    if [ $STOPPED -eq 1 ]; then
      if [ "${FREE:-0}" -ge 45000 ]; then
        (cd ${QI2_ROOT}/page && setsid nohup bash launch_A2_pod1_v1.sh > /dev/null 2>&1 < /dev/null &); sleep 30; STOPPED=0
        log "A2 re-queued: relaunched with --resume at ${FREE} MiB free; PID $(apid)"
      fi
    else
      log "A2 ended by itself: $(tail -1 $ST)"; break
    fi
  elif [ "${FREE:-99999}" -lt 8192 ]; then
    if tr '\0' ' ' < /proc/$P/cmdline | grep -q "page_history_v6i.py.*v6iA2_B2"; then
      kill $P; STOPPED=1; log "GUARD: free ${FREE} MiB below 8192: A2 stopped by PID $P; shares: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
    fi
  fi
  [ $((n % 10)) -eq 0 ] && log "free ${FREE} MiB; shares: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
  n=$((n + 1)); sleep 60
done
