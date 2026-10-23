#!/bin/bash
# Stop the diverged site-level star test (engine_star_v1.py, Stest, tag t1) on the box: PIDs by ps/awk on the exact arguments, never pkill -f.
ps -eo pid,args | awk '/engine_star_v1.py --m 0.2 --g 0.25 --nsites 256 --EF-start 0.95/ && !/awk/ {print $1}' | while read p; do kill "$p" && echo "killed $p"; done
echo "$(date -u +%H:%M:%S) UTC: Stest t1 stopped (site-level Picard loop diverged from pass 0; star families held for the hydrodynamic solver)" >> ${QI2_ROOT}/status.txt
