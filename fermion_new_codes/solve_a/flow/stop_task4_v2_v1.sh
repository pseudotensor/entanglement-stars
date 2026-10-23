#!/bin/bash
# Stops task 4's second launcher (version 3) and its running job by PID (version 4 adds the staggered law's comparison and the first-return
# fraction; every configuration reruns on it).  Builder analyst.
for pid in $(ps -eo pid,args | awk '/bash launch_task4_static_v2\.sh/ && !/awk/ {print $1}'); do kill $pid; done
sleep 1
for pid in $(ps -eo pid,args | awk '/task4_static_v3\.py/ && !/awk/ {print $1}'); do kill $pid; done
date -u "+task4 v2 launcher stopped %H:%M:%S UTC (version 4 reruns every configuration)" >> ${QI2_ROOT}/status.txt
