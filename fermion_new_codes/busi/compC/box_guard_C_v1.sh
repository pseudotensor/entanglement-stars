#!/bin/bash
# The guard of turn 54 46.2 (1) for a C member on a box card: box_guard_C_v1.sh s4|s2 GPU. Every 60 s card GPU's free memory is read (that card
# only); below 8192 MiB the member's process (its pid file, this lane's own) is stopped by that PID; no other process is touched. Every 10 min
# the card's per-process shares are logged.
M=${1:?member}; G=${2:?gpu}; B=${QI2_ROOT}/$M; L=$B/logs/guard.log; n=0
sleep 30; PID=$(cat $B/C32_$M.pid 2>/dev/null)
[ -n "$PID" ] || { echo "$(date -u +%FT%TZ) no pid; guard not started" >> $L; exit 1; }
echo "$(date -u +%FT%TZ) guard start for C32_$M pid $PID on gpu $G (threshold 8192 MiB free)" >> $L
while kill -0 $PID 2>/dev/null; do
  FREE=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${FREE:-99999}" -lt 8192 ]; then
    echo "$(date -u +%FT%TZ) free ${FREE} MiB below 8192 on gpu $G: stopping pid $PID" >> $L; kill $PID; echo "C32_$M STOPPED BY GUARD $(date -u +%FT%TZ) free ${FREE} MiB" >> $B/status.txt; break
  fi
  [ $((n % 10)) -eq 0 ] && echo "$(date -u +%FT%TZ) gpu $G free ${FREE} MiB; used $(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits) MiB" >> $L
  n=$((n+1)); sleep 60
done
echo "$(date -u +%FT%TZ) guard end" >> $L
