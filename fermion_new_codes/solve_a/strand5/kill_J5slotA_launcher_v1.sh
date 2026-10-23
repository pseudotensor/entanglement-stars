#!/bin/bash
# Stops ONLY the slot A launcher's bash process (launch_strand5_J5conv_v7.sh), never its running python child: J5conv5 (star_h0.125_R4)
# continues as an orphaned process and finishes into its own log; the launcher's remaining queue (J5mem2 at h = 1: the star and the
# Gaussian pair) moves to slot C (launch_strand5_J5mem_v4.sh, 4 workers on the threads freed by strand 6 and the production chain), and
# launch_J5conv5_finish_v1.sh writes J5conv5's exit and DONE lines when its log ends.  Signals go to the one PID (not the group).
# ps/awk and PIDs only.  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
if grep -q "star_m0.03_h1_R4 start" J5mem2/status.txt 2>/dev/null; then echo "slot A already reached J5mem2; nothing stopped"; exit 1; fi
for pid in $(ps -u jon -o pid=,args= | awk '$2 == "bash" && $3 == "launch_strand5_J5conv_v7.sh" {print $1}'); do
  kill $pid && echo "stopped launch_strand5_J5conv_v7.sh launcher $pid (its children keep running)"
  date -u "+slot A launcher stopped by the builder %H:%M:%S UTC; star_h0.125_R4 continues; J5mem2 at h = 1 moved to slot C" >> J5conv5/status.txt
done
ps -u jon -o pid=,ppid=,args= | awk '/strand5_w0radial_v10/ && /lstar_m0.1_aH0.5/ && !/awk/ {n++} END {print n " J5conv5 python processes still running"}'
