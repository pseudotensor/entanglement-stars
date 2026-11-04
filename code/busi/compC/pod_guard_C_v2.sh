#!/bin/bash
# The guard of turn 54 46.2 (1) for one C member: pod_guard_C_v2.sh s4|s2 [BASE]. Every 60 s the card's free memory is read; below 8192 MiB the
# member's process (its PID from BASE/C32_<member>.pid, this lane's own) is stopped by that PID, its records kept; no other process is touched.
# Every 10 min both device shares are logged. It ends when the member's process has ended.
M=${1:?member}; B=${2:-${QI2_ROOT}/busi_compC}; L=$B/logs/guard_$M.log; n=0
sleep 30
PID=$(cat $B/C32_$M.pid 2>/dev/null)
[ -n "$PID" ] || { echo "$(date -u +%FT%TZ) no C pid; guard not started" >> $L; exit 1; }
echo "$(date -u +%FT%TZ) guard start for C32_$M pid $PID (threshold 8192 MiB free)" >> $L
while kill -0 $PID 2>/dev/null; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${FREE:-99999}" -lt 8192 ]; then
    echo "$(date -u +%FT%TZ) free ${FREE} MiB below 8192: stopping C32_$M pid $PID (turn 54 46.2 (1)); shares: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> $L
    kill $PID; echo "C32_$M STOPPED BY GUARD $(date -u +%FT%TZ) free ${FREE} MiB" >> $B/status.txt; break
  fi
  if [ $((n % 10)) -eq 0 ]; then echo "$(date -u +%FT%TZ) free ${FREE} MiB; shares: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> $L; fi
  n=$((n+1)); sleep 60
done
echo "$(date -u +%FT%TZ) guard end (C32_$M pid $PID no longer running)" >> $L
