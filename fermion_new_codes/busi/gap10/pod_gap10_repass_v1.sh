#!/bin/bash
# pod_gap10_repass_v1.sh (busi lane): reruns pod_gap10_matrix_v5.sh after each "pass done" (runs deferred for memory) once the card has at
# least 25000 MiB free, at most 4 more passes, until the status file says "all done". The v5 script skips runs logged exit 0. Touches no
# other process; one v5 driver at a time.
ST=${QI2_ROOT}/matrix_status_v5.txt; n=0
while [ $n -lt 4 ]; do
  LAST=$(grep -E "pass done|all done" $ST | tail -1)
  case "$LAST" in "all done"*) echo "repass: all done $(date -u +%H:%M:%S)" >> $ST; exit 0;; esac
  RUNNING=$(ps -eo args | grep -c "[p]od_gap10_matrix_v5.sh")
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  NP=$(grep -c "pass done" $ST)
  if [ "$RUNNING" -eq 0 ] && [ "$NP" -gt "$n" ] && [ "$FREE" -ge 25000 ]; then
    n=$((n+1)); echo "repass $n started $(date -u +%H:%M:%S) (free ${FREE} MiB)" >> $ST
    cd ${QI2_ROOT}/gap10 && bash ${QI2_ROOT}/pod_gap10_matrix_v5.sh >> ${QI2_ROOT}/matrix_v5_nohup.log 2>&1
  fi
  sleep 300
done
