#!/bin/bash
# Stops the collapse runner v2 (bash code/launch_engine_C3_v2.sh code/list_C3_v2.txt), which waits forever: its GPU-free test counted the two
# resident litellm service processes (260 MiB each) on GPU 1 as jobs. It has launched nothing. Exact-argument match; the status line only for a
# process signalled. Builder analyst.
pids=$(ps -eo pid=,args= | awk '$2=="bash" && $3=="code/launch_engine_C3_v2.sh" && $4=="code/list_C3_v2.txt" {print $1}')
if [ -z "$pids" ]; then echo "no C3 v2 runner found"; exit 0; fi
if ps -eo args= | awk '$2=="code/engine_collapse_v4.py"' | grep -q .; then echo "a collapse v4 job is running: not stopping"; exit 1; fi
for p in $pids; do echo "stopping pid $p: $(ps -o args= -p $p)"; kill -TERM $p && echo "C3 runner v2 (pid $p) stopped $(date -u +%H:%M:%S) UTC before launching anything (its GPU-free test counted the resident litellm services on GPU 1)" >> ${QI2_ROOT}/status.txt; done
sleep 1
ps -eo pid=,args= | awk '$2=="bash" && $3=="code/launch_engine_C3_v2.sh"' | grep -q . && echo "STILL RUNNING" || echo "runner gone"
