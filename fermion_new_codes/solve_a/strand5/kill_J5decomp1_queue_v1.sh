#!/bin/bash
# Stops the queued J5decomp1 launcher (launch_strand5_J5decomp_v1.sh, still waiting for J5conv4) and its sleep child, by PID (ps/awk, no pkill);
# spec 10(p''''') redesigns the decomposition (J5decomp2).  Builder analyst.
for pid in $(ps -u jon -o pid=,args= | awk '$2 == "bash" && $3 ~ /launch_strand5_J5decomp_v1\.sh$/ {print $1}'); do
  for c in $(ps -o pid= --ppid $pid); do kill $c 2>/dev/null; done
  kill $pid && echo "stopped launcher $pid"
done
echo "queue stopped by the builder $(date -u +%H:%M:%S) before its first run (spec 10(p''''') redesign: J5decomp2)" >> ${QI2_ROOT}/status.txt
