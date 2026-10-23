#!/bin/bash
# Moves the aware R2 column back to GPU 2 (the CPU benchmark: 153.7 s per t_r = 2 r_s ramp at kappa 8 on 4 threads, i.e. about 7 to 11 hours for
# 16 channels at three ramps, against about 20 minutes on the GPU): stops the CPU run and its launcher subshells, the GPU 2 waiter on list v5 and
# the J45 final waiter v1 (both replaced by versions keyed on list_gpu2_v6.txt). Exact PIDs checked against their arguments. Builder analyst.
chk() { local pid=$1 pat=$2; local a; a=$(ps -o args= -p $pid 2>/dev/null); if [[ "$a" == *"$pat"* ]]; then kill $pid && echo "stopped $pid: ${a:0:100}"; else echo "skip $pid: '${a:0:70}'"; fi; }
chk 2750082 "launch_vf6_cpu_v1.sh"
chk 2750084 "launch_vf6_cpu_v1.sh"
chk 2750090 "engine_vacflow_v6.py --rs 16 --profile exact --m 0 --kmax 16 --r2 --out VF --tag vf6_rs16_exact_m0_R2aware"
chk 2759091 "launch_engine_gpu_v4.sh 2 code/list_gpu2_v5.txt"
chk 2806632 "wait_then_J45final_ell16_v1.sh"
