#!/bin/bash
# The guard of turn 54 46.2 (1) for Computation C on pod 5: every 60 s the card's free memory is read; below 8 GB (8192 MiB) the C process whose
# PID is in ${QI2_ROOT}/C32_s4.pid (this lane's own, launched by pod5_launch_C_s4_v1.sh) is stopped by that PID, its records kept; no other
# process is ever touched. Every 10 min both device shares are printed (nvidia-smi's per-process list). It ends when the C process has ended.
B=${QI2_ROOT}/busi_compC; L=$B/logs/guard.log; n=0
sleep 30
PID=$(cat $B/C32_s4.pid 2>/dev/null)
[ -n "$PID" ] || { echo "$(date -u +%FT%TZ) no C pid; guard not started" >> $L; exit 1; }
echo "$(date -u +%FT%TZ) guard start for C pid $PID (threshold 8192 MiB free)" >> $L
while kill -0 $PID 2>/dev/null; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${FREE:-99999}" -lt 8192 ]; then
    echo "$(date -u +%FT%TZ) free ${FREE} MiB below 8192: stopping C pid $PID (turn 54 46.2 (1)); shares: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> $L
    kill $PID; echo "C32_s4 STOPPED BY GUARD $(date -u +%FT%TZ) free ${FREE} MiB" >> $B/status.txt; break
  fi
  if [ $((n % 10)) -eq 0 ]; then echo "$(date -u +%FT%TZ) free ${FREE} MiB; shares: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> $L; fi
  n=$((n+1)); sleep 60
done
echo "$(date -u +%FT%TZ) guard end (C pid $PID no longer running)" >> $L
