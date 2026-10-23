#!/bin/bash
# Stop the F'''(d) launcher while it waits for J5conv3 (its entropy step would take the 12 workers ahead of J5's convergence and production runs;
# the lead ranks F'''(d) after J5). Only the waiting bash launcher is stopped (ps/awk, by PID); the finished stage-1 and bound records stay.
# Builder analyst.
PIDS=$(ps -u jon -o pid=,args= | awk '/launch_strand3_partFpppd_v1\.sh/ && !/awk/ {print $1}')
echo "killing: $PIDS"
[ -n "$PIDS" ] && kill $PIDS
sleep 1
echo "left eta processes: $(ps -u jon -o pid=,args= | awk '/strand3_partFppp_eta_v2\.py/ && !/awk/' | wc -l)"
echo "queue stopped by the builder $(date -u '+%H:%M:%S') before the entropy step (relaunched behind J5)" >> ${QI2_ROOT}/status.txt
