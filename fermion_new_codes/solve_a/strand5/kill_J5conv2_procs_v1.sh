#!/bin/bash
# Stop the orphaned J5conv2 run (strand5_w0radial_v5.py --out J5conv2, its launcher already stopped) and its pool workers, by PID from ps and
# awk (no pkill/pgrep -f); superseded by J5conv3 (version 6). Builder analyst.
PIDS=$(ps -u jon -o pid=,args= | awk '/strand5_w0radial_v5\.py/ && /--out J5conv2/ && !/awk/ {print $1}')
echo "killing: $PIDS" | tr '\n' ' '; echo
[ -n "$PIDS" ] && kill $PIDS
sleep 2
echo "left: $(ps -u jon -o pid=,args= | awk '/strand5_w0radial_v5\.py/ && /--out J5conv2/ && !/awk/' | wc -l)"
echo "run stopped by the builder $(date -u '+%H:%M:%S') (superseded by J5conv3 with version 6)" >> ${QI2_ROOT}/status.txt
