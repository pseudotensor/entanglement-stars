#!/bin/bash
# Waits for a box GPU with no compute process (nvidia-smi --query-compute-apps empty for its bus id), polling every 5 min until the deadline
# (default 10 h), then runs run_g0pp9c1c_gpu.sh on it. Touches no card that carries a process; records the card picked and nvidia-smi at the pick.
O=${QI2_ROOT}/toy_late_step_v1
C=${QI2_ROOT}/compC
END=$(( $(date +%s) + ${1:-36000} ))
while [ $(date +%s) -lt $END ]; do
  BUSY=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | sort -u)
  FREE=""
  while IFS=, read -r idx bus; do bus=$(echo $bus | tr -d ' '); if ! echo "$BUSY" | grep -q "$bus"; then FREE=$idx; break; fi; done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)
  if [ -n "$FREE" ]; then
    { echo "picked card $FREE at $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,pci.bus_id,memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader; } > $O/g0pp9c1c_gpu.pick
    bash $C/v9c1c/run_g0pp9c1c_gpu.sh $FREE > $O/g0pp9c1c_gpu.log 2>&1; echo "rc $? card $FREE $(date -u +%FT%TZ)" > $O/g0pp9c1c_gpu.done
    exit 0
  fi
  sleep 300
done
echo "deadline: no free card $(date -u +%FT%TZ)" > $O/g0pp9c1c_gpu.done
