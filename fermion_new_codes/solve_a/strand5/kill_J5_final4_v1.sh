#!/bin/bash
# Stops the J5 runs (final4 goal: "J5, P3 and the remainder diagnostics close at their current revisions"): J5len1's pilot_l4_h1 and J5mem2's
# star_m0.03_h0.5_R4 with their launchers and worker pools, by PID (ps/awk only, no pattern kill over ssh).  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
for pid in $(ps -eo pid,args | awk '$2=="bash" && ($3=="code/launch_strand5_J5len1_v2.sh" || $3=="code/launch_strand5_J5mem2_h05_v1.sh") {print $1}'); do echo "kill launcher $pid"; kill $pid; done
for pid in $(ps -eo pid,args | awk '/strand5_w0radial_v1[01].py/ && (/--tag pilot_l4_h1/ || /--tag star_m0.03_h0.5_R4/) && !/awk/ {print $1}'); do echo "kill run $pid"; kill $pid; done
sleep 3
for pid in $(ps -eo pid,ppid,args | awk '$3 ~ /python/ && /strand5_w0radial_v1[01].py/ && !/awk/ {print $1}'); do echo "left: $pid"; done
date -u "+pilot_l4_h1 stopped by the builder %H:%M:%S UTC (final4 goal: J5 closes at its current revision)" >> J5len1/status.txt
date -u "+star_m0.03_h0.5_R4 stopped by the builder %H:%M:%S UTC (final4 goal: J5 closes at its current revision)" >> J5mem2/status.txt
