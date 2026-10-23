#!/bin/bash
# Stops the box3d subagent's list runner loop for list_v3_v1.txt (the bash loop only; its running python child is left to finish), so GPU 2
# passes to J-VF (SPEC_final4b_engine.md sec. 13.6: "GPU 2: J-VF now ... then the subagent's box jobs"). Exact-argument match with ps/awk.
# Builder analyst.
pids=$(ps -eo pid=,args= | awk '$2=="bash" && $3=="code/launch_obs_J45_v1.sh" && $4=="list_v3_v1.txt" {print $1}')
for p in $pids; do echo "stopping list runner pid $p: $(ps -o args= -p $p)"; kill -TERM $p; done
[ -z "$pids" ] && echo "no list runner found"
echo "J45 list_v3_v1 runner stopped by the parent at $(date -u +%Y-%m-%dT%H:%M:%SZ) (GPU 2 to J-VF; the running job left to finish)" >> ${QI2_ROOT}/status.txt
