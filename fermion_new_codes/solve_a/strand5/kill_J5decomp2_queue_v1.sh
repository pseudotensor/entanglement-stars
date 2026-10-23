#!/bin/bash
# Stops the J5decomp2 queue's launcher (launch_strand5_J5decomp_v2.sh) without touching its running python child (the h = 0.5 scan finishes and
# writes its record), and the waiting J5domain1 launcher with its sleep child: the angular-resolution control (J5jres1) runs first.  ps/awk and
# PIDs only.  Builder analyst.
for name in launch_strand5_J5decomp_v2.sh launch_strand5_J5domain_v1.sh; do
  for pid in $(ps -u jon -o pid=,args= | awk -v n="$name" '$2 == "bash" && $3 == n {print $1}'); do
    for c in $(ps -o pid=,comm= --ppid $pid | awk '$2 == "sleep" {print $1}'); do kill $c 2>/dev/null; done
    kill $pid && echo "stopped $name launcher $pid"
  done
done
echo "queue stopped by the builder $(date -u +%H:%M:%S) after gauss_h0.5 (the angular-resolution control J5jres1 first)" >> ${QI2_ROOT}/status.txt
echo "queue stopped by the builder $(date -u +%H:%M:%S) before its first run (replaced by J5jres1)" >> ${QI2_ROOT}/status.txt
