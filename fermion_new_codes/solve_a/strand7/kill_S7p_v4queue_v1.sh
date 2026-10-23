#!/bin/bash
# stop item 7's queue (launch_strand7_S7p_v4.sh: its slot loops) and the running P_j0.5 (strand7_sea_page_v3.py: the GPU SVD makes a refresh about
# 20 s instead of 2); S0_L (static, cached) is left to finish.  By PID from ps; never pkill/pgrep -f.  Builder analyst.
for p in $(ps -eo pid,args | awk "\$2==\"bash\" && \$3==\"launch_strand7_S7p_v4.sh\" {print \$1}"); do kill $p && echo "killed launcher $p"; done
for p in $(ps -eo pid,args | awk "/strand7_sea_page_v3.py/ && /--tag P_j0.5/ && !/awk/ {print \$1}"); do kill $p && echo "killed P_j0.5 $p"; done
date -u "+queue v4 and P_j0.5 stopped by the builder %H:%M:%S UTC (the SVD route is slow; v4 of the code follows; S0_L left to finish)" >> ${QI2_ROOT}/status.txt
