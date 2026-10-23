#!/bin/bash
# Stops the J-VF list runner (bash code/launch_engine_VF_v1.sh code/list_VF_v1.txt) so that no v1 production line starts; its running
# benchmark (python child) is left to finish. Exact-argument match with ps/awk; the status line is written only for a process signalled.
# Builder analyst.
pids=$(ps -eo pid=,args= | awk '$2=="bash" && $3=="code/launch_engine_VF_v1.sh" && $4=="code/list_VF_v1.txt" {print $1}')
if [ -z "$pids" ]; then echo "no J-VF runner found"; exit 0; fi
for p in $pids; do echo "stopping pid $p: $(ps -o args= -p $p)"; kill -TERM $p && echo "J-VF list runner (pid $p) stopped $(date -u +%H:%M:%S) UTC: the production lines move to engine_vacflow_v2 (cost); the running benchmark left to finish" >> ${QI2_ROOT}/status.txt; done
sleep 1
ps -eo pid=,args= | awk '$2=="bash" && $3=="code/launch_engine_VF_v1.sh"' | grep -q . && echo "STILL RUNNING" || echo "runner gone"
ps -eo pid,etimes,args | grep engine_vacflow | grep -v grep | cut -c1-160
