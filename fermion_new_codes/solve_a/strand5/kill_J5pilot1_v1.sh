#!/bin/bash
# Stop the J5 pilot run 1 (strand5_w0radial_v1.py, its averaged centrifugal term has a spurious near-zero mode at large j); builder analyst.
# Finds the launcher and its python children by their command lines with ps and awk (no pkill/pgrep -f), kills by PID.
PIDS=$(ps -u jon -o pid=,args= | awk '/strand5_w0radial_v1\.py/ && /J5pilot1/ && !/awk/ {print $1} /launch_strand5_J5pilot_v1\.sh/ && !/awk/ {print $1}')
echo "killing: $PIDS"
[ -n "$PIDS" ] && kill $PIDS
sleep 2
ps -u jon -o pid=,args= | awk '/strand5_w0radial_v1\.py/ && !/awk/'
echo "stopped by the builder $(date -u '+%H:%M:%S') (spurious radial mode at large j; superseded by version 3)" >> ${QI2_ROOT}/status.txt
