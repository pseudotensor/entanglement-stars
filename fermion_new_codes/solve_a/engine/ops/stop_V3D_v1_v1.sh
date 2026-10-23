#!/bin/bash
# Stop V3D version 1 (launch_engine_V3D_v1.sh and its running case): version 1 multiplies dense 13500 x 13500 by 13500 x 6750 on one CPU
# thread (about 30 minutes per L = 15 case, GPU idle); version 2 (sparse products, identical sums to 9e-16 on L11_s3_B0.3_ho0) reruns the
# L = 15 cases.  PIDs by ps/awk on the exact arguments, never pkill -f.
ps -eo pid,args | awk '/bash launch_engine_V3D_v1.sh/ && !/awk/ {print $1}' | while read p; do kill "$p" && echo "killed launcher $p"; done
ps -eo pid,args | awk '/engine_validate3d_v1.py --L 15/ && !/awk/ {print $1}' | while read p; do kill "$p" && echo "killed case $p"; done
echo "$(date -u +%H:%M:%S) UTC: V3D v1 stopped at L15_s3_B0_ho0 (CPU-bound dense products); L = 15 reruns on version 2" >> ${QI2_ROOT}/status.txt
