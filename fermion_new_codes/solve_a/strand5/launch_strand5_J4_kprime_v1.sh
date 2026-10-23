#!/bin/bash
# Strand 5 J4 stage 2, sec. 10(j)(7): the single-packet coefficient k(m) at m_s (1 -+ 1e-3) for the three distinct masses of each J4 point
# ((0.1, 1), (0.1, 0.5), (0.03, 0.3), (0.01, 0.1); twist 0, L = 256) by the J2b routine strand5_kernel_packets_v1.py (set C, masses given),
# 8 single-thread workers.  Output s5_star/J4kprime_run1.
D=${QI2_ROOT}/s5_star
cd $D || exit 1
O=J4kprime_run1
mkdir -p $O
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_kernel_packets_v1.py --out $O --set C --L 256 --workers 8 --masses 0.099900000000000003,0.10009999999999999,1.0039825745499769,1.0059925496742008,1.4163269431879066,1.419162432563658,0.099900000000000003,0.10009999999999999,0.50939204940791927,0.51041185331063776,0.71342870001143066,0.71485698569713918,0.02997,0.030029999999999994,0.30119477236499309,0.30179776490226029,0.42489808295637199,0.42574872976909744,0.0099900000000000006,0.01001,0.1003982574549977,0.1005992549674201,0.14163269431879069,0.14191624325636584 > $O/run.log 2>&1
echo "exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
echo DONE >> $O/status.txt
