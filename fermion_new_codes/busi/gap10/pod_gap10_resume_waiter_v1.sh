#!/bin/bash
# pod_gap10_resume_waiter_v1.sh (busi lane): restarts pod_gap10_matrix_v5.sh (stopped by this lane at about 00:22 UTC on 2026-10-18 for room)
# once C's coupling-off control (pid file ${QI2_ROOT}/C32_s4off.pid) uses at least 20000 MiB (past its preparation) and the card has at
# least 36000 MiB free (a GAP10 run's 28 GB plus the script's 6 GB margin), seen on 2 checks 5 minutes apart. The v5 script skips exit-0 runs.
# If C's control ends first, it starts when 36000 MiB are free. Reads nvidia-smi and the pid file only; touches no other process.
ST=${QI2_ROOT}/matrix_status_v5.txt; ok=0
while true; do
  CP=$(cat ${QI2_ROOT}/C32_s4off.pid 2>/dev/null)
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  CUSE=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$CP" '$1==p{print $2}')
  CUSE=${CUSE:-0}
  if kill -0 "$CP" 2>/dev/null; then cond=$([ "$CUSE" -ge 20000 ] && [ "$FREE" -ge 36000 ] && echo 1 || echo 0); else cond=$([ "$FREE" -ge 36000 ] && echo 1 || echo 0); fi
  if [ "$cond" = 1 ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 2 ]; then
    echo "v5 RESUMED by the waiter at $(date -u +%H:%M:%S) (free ${FREE} MiB, C s4off ${CUSE} MiB)" >> $ST
    cd ${QI2_ROOT}/gap10 && setsid nohup bash ${QI2_ROOT}/pod_gap10_matrix_v5.sh >> ${QI2_ROOT}/matrix_v5_nohup.log 2>&1 < /dev/null &
    exit 0
  fi
  sleep 300
done
