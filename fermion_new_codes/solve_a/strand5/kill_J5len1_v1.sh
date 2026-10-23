#!/bin/bash
# stop J5len1 version 1 (its j count 4n + 4 exceeds the code's 4n: the last channels start beyond the grid) and its launcher, by PID; the
# pool workers carry the same command line.  Builder analyst.
for p in $(ps -eo pid,args | awk "\$2==\"bash\" && \$3==\"launch_strand5_J5len1_v1.sh\" {print \$1}"); do kill $p && echo "killed launcher $p"; done
for p in $(ps -eo pid,args | awk "/strand5_w0radial_v11.py/ && /--tag pilot_l/ && !/awk/ {print \$1}"); do kill $p && echo "killed $p"; done
date -u "+J5len1 version 1 stopped by the builder %H:%M:%S UTC (j count 4n + 4 above the code limit 4n; version 2 uses 4n)" >> ${QI2_ROOT}/status.txt
