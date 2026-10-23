#!/bin/bash
# Stops the GPU 2 waiter on list_gpu2_v4.txt (its two vf6 lines now run on the CPU, launch_vf6_cpu_v1.sh); replaced by list_gpu2_v5.txt.
# Exact PID checked against its arguments. Builder analyst.
pid=2404634
a=$(ps -o args= -p $pid 2>/dev/null)
if [[ "$a" == *"launch_engine_gpu_v4.sh 2 code/list_gpu2_v4.txt"* ]]; then kill $pid && echo "stopped $pid: $a"; else echo "skip $pid: '$a'"; fi
