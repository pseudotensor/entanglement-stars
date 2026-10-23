#!/bin/bash
# Stops task 4's first launcher and its running job by PID (version 3 lengthens the run for the interior-entering packets; every configuration
# reruns on it).  Builder analyst.
for pid in $(ps -eo pid,args | awk '/bash launch_task4_static_v1\.sh/ && !/awk/ {print $1}'); do kill $pid; done
sleep 1
for pid in $(ps -eo pid,args | awk '/task4_static_v2\.py/ && !/awk/ {print $1}'); do kill $pid; done
date -u "+task4 v1 launcher stopped %H:%M:%S UTC (version 3 reruns every configuration)" >> ${QI2_ROOT}/status.txt
