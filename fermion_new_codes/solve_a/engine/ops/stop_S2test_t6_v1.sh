#!/bin/bash
# Stop the star test t6 (engine_star_v3 before the occupied-columns change: the sources multiplied every positive mode on one CPU thread,
# about 30 s per pass); rerun as t7.  PIDs by ps/awk on the exact arguments.
ps -eo pid,args | awk '/engine_star_v3.py --m 0.1 --g 0.25 --ell 8 --filling zeroT_NB --NB-list 801/ && /--tag t6/ && !/awk/ {print $1}' | while read p; do kill "$p" && echo "killed $p"; done
echo "$(date -u +%H:%M:%S) UTC: S2test t6 stopped (sources over all positive modes); rerun as t7" >> ${QI2_ROOT}/status.txt
