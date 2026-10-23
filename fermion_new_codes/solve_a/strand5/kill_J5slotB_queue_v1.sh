#!/bin/bash
# Stops the waiting J5 slot B launcher (launch_strand5_J5mem_v1.sh, 8 workers) and its sleep child before its first run: version 2 runs 6 workers
# so that strand 6's two-thread chain fits the 16-thread budget.  ps/awk and PIDs only.  Builder analyst.
for pid in $(ps -u jon -o pid=,args= | awk '$2 == "bash" && $3 == "launch_strand5_J5mem_v1.sh" {print $1}'); do
  for c in $(ps -o pid=,comm= --ppid $pid | awk '$2 == "sleep" {print $1}'); do kill $c 2>/dev/null; done
  kill $pid && echo "stopped launch_strand5_J5mem_v1.sh launcher $pid"
done
