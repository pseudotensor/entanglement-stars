#!/bin/bash
# Stops the waiting J5 slot B launcher (launch_strand5_J5mem_v2.sh) and its sleep child before its first run: version 3 also waits for strand 7's
# S7a chain (six single-thread jobs on slot B's share; the lead's order puts J5mem2 h = 0.5 last, yielding threads).  Refuses if the h = 0.5 run
# has started.  ps/awk and PIDs only.  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
if grep -q "star_m0.03_h0.5_R4 start" J5mem2/status.txt 2>/dev/null; then echo "h = 0.5 run already started; nothing stopped"; exit 1; fi
for pid in $(ps -u jon -o pid=,args= | awk '$2 == "bash" && $3 == "launch_strand5_J5mem_v2.sh" {print $1}'); do
  for c in $(ps -o pid=,comm= --ppid $pid | awk '$2 == "sleep" {print $1}'); do kill $c 2>/dev/null; done
  kill $pid && echo "stopped launch_strand5_J5mem_v2.sh launcher $pid"
done
