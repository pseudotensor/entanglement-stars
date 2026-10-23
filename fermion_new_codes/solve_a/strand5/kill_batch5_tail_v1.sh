#!/bin/bash
# stop batch 5's waiting tail (launch_strand5_J4_stage2_v5.sh, waiting for J4run4) and the cons v4 waiter (waiting for batch 5's DONE), by PID
for pat in "launch_strand5_J4_stage2_v5.sh" "launch_strand5_J4_cons_v4.sh"; do
  for pid in $(ps -u jon -o pid=,args= | awk -v p="$pat" 'index($0, p) && !index($0, "awk") {print $1}'); do echo "kill $pid ($pat)"; kill $pid; done
done
sleep 1; ps -u jon -o pid=,args= | awk '(index($0, "stage2_v5.sh") || index($0, "cons_v4.sh")) && !index($0, "awk")' | cut -c1-100
