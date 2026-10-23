#!/bin/bash
# stop the waiting S7p launcher (launch_strand7_S7p_v1.sh) by PID; it has started no runs (it waits for J5conv5)
for p in $(ps -eo pid,args | awk "\$2==\"bash\" && \$3==\"launch_strand7_S7p_v1.sh\" {print \$1}"); do kill $p && echo "killed $p"; done
date -u "+launcher v1 stopped by the builder %H:%M:%S UTC (P to the background end 30 r_s0 in v2)" >> ${QI2_ROOT}/status.txt
