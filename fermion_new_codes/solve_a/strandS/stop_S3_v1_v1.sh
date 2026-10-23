#!/bin/bash
# Stops strand S job S3's version 1 launcher and its workers by PID (the lead's sec. 9.5 rulings change the ladder and add the two-gap check;
# version 2 reruns every normal; version 1's [100] record is kept).  Builder analyst.
S=${QI2_ROOT}/sS_entropy
for pid in $(ps -eo pid,args | awk '/bash code\/launch_strandS_S3_v1\.sh/ && !/awk/ {print $1}'); do echo "stopping launcher $pid"; kill $pid; done
sleep 1
for pid in $(ps -eo pid,args | awk '/strandS_sphere_v1\.py/ && !/awk/ {print $1}'); do kill $pid; done
date -u "+S3 v1 stopped %H:%M:%S UTC (the lead's rulings sec. 9.5; v2 reruns every normal; the [100] record kept)" >> $S/S3/status.txt
