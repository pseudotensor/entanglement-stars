#!/bin/bash
# stop the J4 stage-2 version-1 batch: first its xargs (so no new workers start), then the python workers, by PID
for pid in $(ps -u jon -o pid=,args= | awk 'index($0, "xargs -P 8 -L 1 bash -c run") && !index($0, "awk") {print $1}'); do
  echo "kill xargs $pid"; kill $pid
done
sleep 1
for pid in $(ps -u jon -o pid=,args= | awk 'index($0, "strand5_J4_stage2_v1.py") && !index($0, "awk") {print $1}'); do
  echo "kill $pid $(ps -o args= -p $pid | cut -c1-110)"; kill $pid
done
sleep 2
echo "remaining: $(ps -u jon -o args= | grep -c 'strand5_J4_stage2_v1.py\|xargs -P 8 -L 1 bash -c run')"
