#!/bin/bash
# Stop the queued (still waiting, not computing) J5 convergence launcher so it can be relaunched behind the 10(r)(b) second-crossing pass
# (thread budget); finds it with ps and awk, kills by PID. Builder analyst.
PIDS=$(ps -u jon -o pid=,args= | awk '/launch_strand5_J5conv_v1\.sh/ && !/awk/ {print $1}')
echo "killing: $PIDS"
[ -n "$PIDS" ] && kill $PIDS
sleep 1
ps -u jon -o pid=,args= | awk '/strand5_w0radial_v5\.py/ && /J5conv1/ && !/awk/'
echo "queue stopped by the builder $(date -u '+%H:%M:%S') (relaunched behind the second-crossing pass)" >> ${QI2_ROOT}/status.txt
