#!/bin/bash
# stop the J4 stage-2 version-1 batch (launcher, xargs and its python workers) by PID; only processes whose command line names them
for pat in launch_strand5_J4_stage2_v1.sh strand5_J4_stage2_v1.py; do
  for pid in $(ps -u jon -o pid=,args= | awk -v p="$pat" 'index($0, p) && !index($0, "awk") && !index($0, "kill_J4s2run1") {print $1}'); do
    echo "kill $pid $(ps -o args= -p $pid | cut -c1-120)"; kill $pid
  done
done
sleep 2
ps -u jon -o pid=,args= | grep -c "strand5_J4_stage2_v1" 
