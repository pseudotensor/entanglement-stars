#!/bin/bash
# launch_page_history_v6_wait_v1.sh (builder analyst, 2026-10-16; the lead's lane 2c request): waits for the saved state of A s4 that a run
# starts from, then for free device memory, then runs code_v6/launch_page_history_v6_v3.sh <name> (its own gates follow: the history
# oracle v6c, du -sm /workspace below 40000 MB, the container disk). The state is complete when its final path exists: page_history_v6b's
# save_state writes <path>.tmp and renames it (os.replace). The device gate: the free memory (nvidia-smi, the pod's one GPU) at least
# NEED_GPU_GB (default 45: C's estimated peak of about 27 GB plus A s4's swing of about 12 GB between its records and its checkpoints, with
# a margin) on three polls 20 s apart, so the run does not start beside the lane 5 regression's runs. Both waits end at WAIT_H hours
# (default 4) with exit 7 (the state) or 8 (the memory), nothing launched. Usage: bash code_v6/launch_page_history_v6_wait_v1.sh c15 | c30 | s4tr | <name>
set -euo pipefail
cd ${QI2_ROOT}/page
name=${1:?usage: launch_page_history_v6_wait_v1.sh <name>}
need=${NEED_GPU_GB:-45}
wait_h=${WAIT_H:-4}
now() { date -u +%H:%M:%S; }
case "$name" in
  c15|s4tr) st=${QI2_ROOT}/v6bA_s4_state_at_015.000.pt;;
  c30) st=${QI2_ROOT}/v6bA_s4_state_at_030.000.pt;;
  *) st="";;
esac
deadline=$(( $(date +%s) + wait_h * 3600 ))
if [ -n "$st" ]; then
  echo "$(now) UTC: $name waits for the state $st (complete when the final path exists)"
  until [ -f "$st" ]; do
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "$(now) UTC: the state $st did not appear within $wait_h h: no launch"; exit 7; fi
    sleep 60
  done
  echo "$(now) UTC: the state $st is complete ($(stat -c %s "$st") bytes)"
fi
ok=0; free=0
while :; do
  line=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  tot=${line%,*}; used=${line#*,}
  free=$(( (tot - used) / 1024 ))
  if [ "$free" -ge "$need" ]; then ok=$((ok + 1)); else ok=0; fi
  [ "$ok" -ge 3 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "$(now) UTC: the device's free memory stayed below $need GB (last $free GB): no launch"; exit 8; fi
  sleep 20
done
echo "$(now) UTC: the device has $free GB free (need $need GB on three polls 20 s apart): launching $name"
exec bash code_v6/launch_page_history_v6_v3.sh "$name"
