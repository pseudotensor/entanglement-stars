#!/bin/bash
# Moves the collapse runs to engine_collapse_v9 (fields v7's floor-aware convergence; the pair block exported): stops, by exact PIDs checked
# against their arguments, the GPU 1 runner and its v8 primary, the GPU 0 runner and its v7 ell-8 control, the GPU 2 v3-list waiter (its list
# had not started; the v2 runner with the J-VF jobs is kept), and the J3' v4 waiter (v8 paths). Runners first, then their children.
# Builder analyst.
chk() { local pid=$1 pat=$2; local a; a=$(ps -o args= -p $pid 2>/dev/null); if [[ "$a" == *"$pat"* ]]; then kill $pid && echo "stopped $pid: ${a:0:110}"; else echo "skip $pid: '${a:0:80}' does not match '$pat'"; fi; }
chk 2260899 "launch_engine_gpu_v4.sh 1 code/list_gpu1_v8.txt"
chk 1318314 "launch_engine_gpu_v3.sh 0 code/list_gpu0_v2.txt"
chk 1408408 "launch_engine_gpu_v3.sh 2 code/list_gpu2_v3.txt"
chk 2337823 "launch_obs_J3v3_primary_v4.sh"
chk 2268400 "engine_collapse_v8.py --m 0.2 --g 0.25 --rs 64"
chk 1714019 "engine_collapse_v7.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 8"
