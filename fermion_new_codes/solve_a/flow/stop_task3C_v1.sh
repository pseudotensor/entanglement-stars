#!/bin/bash
# Stops task 3's cut-table queue (launch_flow_task3_C_v1.sh) and its workers by PID: its gate and S3's raced at 04:50:44 UTC and the lane went
# above 16 threads; the tables are requeued behind S3.  Builder analyst.
for pid in $(ps -eo pid,args | awk '/bash launch_flow_task3_C_v1\.sh/ && !/awk/ {print $1}'); do kill $pid; done
sleep 1
for pid in $(ps -eo pid,args | awk '/flow_chart_v4\.py cut/ && !/awk/ {print $1}'); do kill $pid; done
date -u "+task3C stopped %H:%M:%S UTC (the thread cap; requeued behind S3)" >> ${QI2_ROOT}/status.txt
