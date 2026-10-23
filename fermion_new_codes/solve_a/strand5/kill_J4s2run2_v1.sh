#!/bin/bash
# stop the J4 stage-2 version-2 batch (J4s2run2): its launcher, its xargs, then the python workers, by PID
for pat in "launch_strand5_J4_stage2_v2.sh" "xargs -P 8 -L 1 bash -c run"; do
  for pid in $(ps -u jon -o pid=,args= | awk -v p="$pat" 'index($0, p) && !index($0, "awk") {print $1}'); do echo "kill $pid ($pat)"; kill $pid; done
done
sleep 1
for pid in $(ps -u jon -o pid=,args= | awk 'index($0, "strand5_J4_stage2_v2.py") && index($0, "J4s2run2") && !index($0, "awk") {print $1}'); do
  echo "kill $pid $(ps -o args= -p $pid | cut -c1-100)"; kill $pid
done
sleep 2
ps -u jon -o pid=,args= | awk 'index($0, "stage2_v2") && !index($0, "awk")' | cut -c1-120
