#!/bin/bash
# Replaces GPU 0's runner on list_gpu0_v3.txt by one on list_gpu0_v4.txt (engine 17.6: V3D with the chain-current variant after the ell = 8
# control): stops the runner's bash only; its child, the running ell = 8 control (engine_collapse_v9), continues and the new runner waits for it.
# Exact PID checked against its arguments. Builder analyst.
pid=2404633
a=$(ps -o args= -p $pid 2>/dev/null)
if [[ "$a" == *"launch_engine_gpu_v4.sh 0 code/list_gpu0_v3.txt"* ]]; then kill $pid && echo "stopped $pid: $a"; else echo "skip $pid: '$a'"; fi
ps -o pid,args= -p 2412342 | cut -c1-100
