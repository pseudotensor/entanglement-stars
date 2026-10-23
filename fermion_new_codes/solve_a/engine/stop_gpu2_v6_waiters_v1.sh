#!/bin/bash
# Replaces GPU 2's waiter on list_gpu2_v6.txt (v9 runs) by list_gpu2_v7.txt (the r_s = 32 set on engine_collapse_v10, engine 17.6) and the J45
# final waiter v2 by v3 (keyed on list v7, with the post list for C5.3' and J6). Exact PIDs checked against their arguments. Builder analyst.
chk() { local pid=$1 pat=$2; local a; a=$(ps -o args= -p $pid 2>/dev/null); if [[ "$a" == *"$pat"* ]]; then kill $pid && echo "stopped $pid: ${a:0:100}"; else echo "skip $pid: '${a:0:70}'"; fi; }
chk 3792557 "launch_engine_gpu_v4.sh 2 code/list_gpu2_v6.txt"
chk 3792558 "wait_then_J45final_ell16_v2.sh"
