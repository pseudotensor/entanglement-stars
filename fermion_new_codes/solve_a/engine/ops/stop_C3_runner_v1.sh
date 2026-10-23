#!/bin/bash
# Stops the collapse runner (bash code/launch_engine_C3_v1.sh code/list_C3_v1.txt), still waiting for J3 (it has launched nothing), so the
# list moves to engine_collapse_v4 (the surface's Taylor records and r_in of SPEC_final4b_observables.md sec. 10.2 B1). Exact-argument match;
# the status line is written only for a process signalled. Builder analyst.
pids=$(ps -eo pid=,args= | awk '$2=="bash" && $3=="code/launch_engine_C3_v1.sh" && $4=="code/list_C3_v1.txt" {print $1}')
if [ -z "$pids" ]; then echo "no C3 runner found"; exit 0; fi
if ps -eo args= | awk '$2=="code/engine_collapse_v3.py"' | grep -q .; then echo "a collapse v3 job is running: not stopping"; exit 1; fi
for p in $pids; do echo "stopping pid $p: $(ps -o args= -p $p)"; kill -TERM $p && echo "C3 runner (pid $p) stopped $(date -u +%H:%M:%S) UTC before launching anything: the list moves to engine_collapse_v4" >> ${QI2_ROOT}/status.txt; done
sleep 1
ps -eo pid=,args= | awk '$2=="bash" && $3=="code/launch_engine_C3_v1.sh"' | grep -q . && echo "STILL RUNNING" || echo "runner gone"
