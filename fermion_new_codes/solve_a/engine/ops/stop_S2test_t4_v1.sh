#!/bin/bash
# Stop the star test t4 (engine_star_v2, linear mixing, about 10 percent per pass); rerun on engine_star_v3 (Anderson mixing; the same fixed
# point: t5 reproduced t3's E 19.468640 in 19 passes against 172).  PIDs by ps/awk on the exact arguments.
ps -eo pid,args | awk '/engine_star_v2.py --m 0.1 --g 0.25 --ell 8 --filling zeroT_NB --NB-list 801/ && !/awk/ {print $1}' | while read p; do kill "$p" && echo "killed $p"; done
echo "$(date -u +%H:%M:%S) UTC: S2test t4 stopped (linear mixing); rerun as t6 on engine_star_v3 (Anderson)" >> ${QI2_ROOT}/status.txt
