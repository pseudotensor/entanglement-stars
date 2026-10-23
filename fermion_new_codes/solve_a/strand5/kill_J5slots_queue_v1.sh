#!/bin/bash
# Stops the two waiting J5 slot launchers (launch_strand5_J5conv_v5.sh, launch_strand5_J5member_v1.sh) and their sleep children before their first
# run: spec 10(p'''''') names the member's folder J5mem2 and adds the flat control and the member's Gaussian pair.  ps/awk and PIDs only.  Builder analyst.
for name in launch_strand5_J5conv_v5.sh launch_strand5_J5member_v1.sh; do
  for pid in $(ps -u jon -o pid=,args= | awk -v n="$name" '$2 == "bash" && $3 == n {print $1}'); do
    for c in $(ps -o pid=,comm= --ppid $pid | awk '$2 == "sleep" {print $1}'); do kill $c 2>/dev/null; done
    kill $pid && echo "stopped $name launcher $pid"
  done
done
cd ${QI2_ROOT}/s5_star && echo "queue stopped by the builder $(date -u +%H:%M:%S) before its first run (spec 10(p''''''): flat control, J5mem2)" >> J5conv5/status.txt
rmdir J5member1 2>/dev/null; ls -d J5member1 2>/dev/null && echo "J5member1 not empty, kept"
