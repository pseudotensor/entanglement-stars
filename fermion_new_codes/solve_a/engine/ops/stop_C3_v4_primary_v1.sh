#!/bin/bash
# Stops the engine_collapse_v4 primary (SPEC_final4b_engine.md sec. 15.4: "stop the v4 primary; ... keep v4's record as the spline comparator";
# the site fields move to the dense solution, engine_collapse_v5). Exact-argument match on the python process; the runner v3 then reaches its
# end. The status line only for a process signalled. Builder analyst.
pids=$(ps -eo pid=,args= | awk '$3=="code/engine_collapse_v4.py" {for (i=1;i<=NF;i++) if ($i=="--tag" && $(i+1)=="primary_rs64_A_R2_tau4") print $1}')
if [ -z "$pids" ]; then echo "no v4 primary found"; exit 0; fi
for p in $pids; do echo "stopping pid $p: $(ps -o args= -p $p | cut -c1-150)"; kill -TERM $p && echo "C3 primary_rs64_A_R2_tau4 (engine_collapse_v4, pid $p) stopped $(date -u +%H:%M:%S) UTC by the lead's sec. 15.1 (dense site fields); its JSON record kept as the spline comparator" >> ${QI2_ROOT}/status.txt; done
sleep 2
ps -eo args= | awk '$2=="code/engine_collapse_v4.py"' | grep -q . && echo "STILL RUNNING" || echo "v4 primary gone"
