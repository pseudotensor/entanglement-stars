#!/bin/bash
# Stop the queued (still waiting, not computing) J5 convergence launcher so it can be relaunched behind the 10(r)(b) second-crossing pass
# (thread budget); finds it with ps and awk, kills by PID. Builder analyst.
PIDS=$(ps -u jon -o pid=,args= | awk '/launch_strand5_J5conv_v2\.sh/ && !/awk/ {print $1}')
echo "killing: $PIDS"
[ -n "$PIDS" ] && kill $PIDS
sleep 1
ps -u jon -o pid=,args= | awk '/strand5_w0radial_v5\.py/ && /J5conv2/ && !/awk/'
echo "queue stopped by the builder $(date -u '+%H:%M:%S') (relaunched as version 3 with strand5_w0radial_v6.py and the exact-L linear response)" >> ${QI2_ROOT}/status.txt
