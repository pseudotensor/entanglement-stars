#!/bin/bash
# Stops the waiting P1 trio launcher (launch_strand5_P1trio_v1.sh) and its sleep child: the production chain's two threads are free now
# (P1, P3 and P5 done), so the trio addendum runs there directly.  Refuses if the trio has run.  ps/awk and PIDs only.  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
if grep -q "P1 trio exit" P1run1/status.txt 2>/dev/null; then echo "trio already ran; nothing stopped"; exit 1; fi
for pid in $(ps -u jon -o pid=,args= | awk '$2 == "bash" && $3 == "launch_strand5_P1trio_v1.sh" {print $1}'); do
  for c in $(ps -o pid=,comm= --ppid $pid | awk '$2 == "sleep" {print $1}'); do kill $c 2>/dev/null; done
  kill $pid && echo "stopped launch_strand5_P1trio_v1.sh launcher $pid"
done
