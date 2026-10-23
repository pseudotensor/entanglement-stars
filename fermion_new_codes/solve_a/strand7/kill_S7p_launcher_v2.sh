#!/bin/bash
# stop the waiting S7p launcher version 2 (it has started no runs: it waits for J5conv5) by PID; version 3 gates on free threads instead.
for p in $(ps -eo pid,args | awk "\$2==\"bash\" && \$3==\"launch_strand7_S7p_v2.sh\" {print \$1}"); do kill $p && echo "killed $p"; done
date -u "+launcher v2 stopped by the builder %H:%M:%S UTC (v3 gates each run on the lane's free threads)" >> ${QI2_ROOT}/status.txt
