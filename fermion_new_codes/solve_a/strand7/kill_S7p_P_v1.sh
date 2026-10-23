#!/bin/bash
# stop item 7's P runs on the held-lock background (the lead's turn-5 ruling withdraws it: SPEC_strand7_page.md sec. 12) and their launcher v3,
# by PID from ps (never pkill/pgrep -f).  Builder analyst.
for p in $(ps -eo pid,args | awk "\$2==\"bash\" && \$3==\"launch_strand7_S7p_v3.sh\" {print \$1}"); do kill $p && echo "killed launcher $p"; done
for p in $(ps -eo pid,args | awk "/strand7_sea_page_v1.py/ && /--tag P_j/ && !/awk/ {print \$1}"); do ps -o pid=,args= -p $p | cut -c1-160; kill $p && echo "killed $p"; done
date -u "+P runs and launcher v3 stopped by the builder %H:%M:%S UTC (the lead withdrew the held-lock background, page spec sec. 12)" >> ${QI2_ROOT}/status.txt
