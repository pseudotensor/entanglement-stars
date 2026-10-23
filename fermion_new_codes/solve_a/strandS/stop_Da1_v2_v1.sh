#!/bin/bash
# Stops D-a1's version 2 run (E16; the third radius of version 3 supersedes it; version 2's E8 record is kept) by PID.  Builder analyst.
for pid in $(ps -eo pid,args | awk '/bash code\/launch_strandS_Da1_v1\.sh/ && !/awk/ {print $1}'); do kill $pid; done
for pid in $(ps -eo pid,args | awk '/strandS_dos_v2\.py/ && !/awk/ {print $1}'); do echo "stopping $pid"; kill $pid; done
date -u "+Da1 v2 stopped %H:%M:%S UTC (version 3 adds the third radius; v2's E8 kept)" >> ${QI2_ROOT}/status.txt
