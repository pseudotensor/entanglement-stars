#!/bin/bash
# Stops the waiting J5 slot A launcher (launch_strand5_J5conv_v6.sh) and their sleep children before their first
# run: version 7 runs 6 workers so that the production chain keeps 2 threads.  ps/awk and PIDs only.  Builder analyst.
for name in launch_strand5_J5conv_v6.sh; do
  for pid in $(ps -u jon -o pid=,args= | awk -v n="$name" '$2 == "bash" && $3 == n {print $1}'); do
    for c in $(ps -o pid=,comm= --ppid $pid | awk '$2 == "sleep" {print $1}'); do kill $c 2>/dev/null; done
    kill $pid && echo "stopped $name launcher $pid"
  done
done
cd ${QI2_ROOT}/s5_star && echo "queue stopped by the builder $(date -u +%H:%M:%S) before its first run (slot A relaunched on 6 workers: launch_strand5_J5conv_v7.sh)" >> J5conv5/status.txt
