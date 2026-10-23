#!/bin/bash
# stop the cross-check waiter of batch 5 (J4consrun3, cons version 1), by PID
for pat in "launch_strand5_J4_cons_v3.sh"; do
  for pid in $(ps -u jon -o pid=,args= | awk -v p="$pat" 'index($0, p) && !index($0, "awk") {print $1}'); do echo "kill $pid ($pat)"; kill $pid; done
done
for pid in $(ps -u jon -o pid=,args= | awk 'index($0, "xargs -P 4 -L 1 bash -c run") && !index($0, "awk") {print $1}'); do ps -o args= -p $pid | grep -q . && echo "kill xargs $pid" && kill $pid; done
for pid in $(ps -u jon -o pid=,args= | awk 'index($0, "strand5_J4_stage2_cons_v1.py") && !index($0, "awk") {print $1}'); do echo "kill $pid"; kill $pid; done
sleep 1
ps -u jon -o pid=,args= | awk '(index($0, "cons_v3.sh") || index($0, "cons_v1.py")) && !index($0, "awk")' | cut -c1-100
