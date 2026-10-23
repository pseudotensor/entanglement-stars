#!/bin/bash
# stop the J4 stage-2 batch 4 (launcher, xargs, workers) and its cross-check waiter (launcher, xargs, waiting shells), by PID
for pat in "launch_strand5_J4_stage2_v4.sh" "launch_strand5_J4_cons_v2.sh" "xargs -P 12 -L 1 bash -c run" "xargs -P 4 -L 1 bash -c run"; do
  for pid in $(ps -u jon -o pid=,args= | awk -v p="$pat" 'index($0, p) && !index($0, "awk") {print $1}'); do echo "kill $pid ($pat)"; kill $pid; done
done
sleep 1
for pid in $(ps -u jon -o pid=,args= | awk '(index($0, "strand5_J4_stage2_v3.py") || index($0, "strand5_J4_stage2_cons_v1.py")) && !index($0, "awk") {print $1}'); do echo "kill $pid"; kill $pid; done
for pid in $(ps -u jon -o pid=,args= | awk 'index($0, "bash -c run") && !index($0, "awk") && !index($0, "xargs") {print $1}'); do kill $pid 2>/dev/null && echo "kill shell $pid"; done
sleep 2
echo "remaining:"; ps -u jon -o pid=,args= | awk '(index($0, "stage2_v3") || index($0, "J4_cons") || index($0, "bash -c run") || index($0, "stage2_v4.sh")) && !index($0, "awk")' | cut -c1-120
