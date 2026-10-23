#!/bin/bash
# Stops the box3d subagent's waiter wait_then_v3b_v2.sh (it would start list_v3b_v1.txt on GPU 2 when smallball3 ends); GPU 2 goes to J-VF
# first (SPEC_final4b_engine.md sec. 13.6). The waiter has launched nothing; smallball3 is left to finish. Exact-argument match with ps/awk;
# the status line is written only for a process actually signalled. Builder analyst.
pids=$(ps -eo pid=,args= | awk '$2=="bash" && $3=="wait_then_v3b_v2.sh" {print $1}')
if [ -z "$pids" ]; then echo "no waiter found"; exit 0; fi
for p in $pids; do echo "stopping waiter pid $p: $(ps -o args= -p $p)"; kill -TERM $p && echo "J45 waiter wait_then_v3b_v2.sh (pid $p) stopped by the parent $(date -u +%Y-%m-%dT%H:%M:%SZ): GPU 2 to J-VF first; list_v3b_v1.txt kept for the hand-back" >> ${QI2_ROOT}/status.txt; done
sleep 1
ps -eo pid=,args= | awk '$2=="bash" && $3=="wait_then_v3b_v2.sh"' | grep -q . && echo "STILL RUNNING" || echo "waiter gone"
