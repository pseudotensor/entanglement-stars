#!/bin/bash
# Stops the waiting strand 6 (i) launcher (launch_strand6i_v1.sh) and its sleep child before its run: the production chain's two threads
# are free (P1, P3, P5 done), so version 2 starts (i) there now instead of at J5decomp3's end.  Refuses if (i) has started.
# ps/awk and PIDs only.  Builder analyst.
cd ${QI2_ROOT}/s6_val || exit 1
if grep -q "^start" s6i_run1/status.txt 2>/dev/null; then echo "(i) already started; nothing stopped"; exit 1; fi
for pid in $(ps -u jon -o pid=,args= | awk '$2 == "bash" && $3 == "launch_strand6i_v1.sh" {print $1}'); do
  for c in $(ps -o pid=,comm= --ppid $pid | awk '$2 == "sleep" {print $1}'); do kill $c 2>/dev/null; done
  kill $pid && echo "stopped launch_strand6i_v1.sh launcher $pid"
done
