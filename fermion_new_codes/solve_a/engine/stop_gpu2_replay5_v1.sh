#!/bin/bash
# stop_gpu2_replay5_v1.sh (builder analyst): the ell 8 control's replay (engine_replay_v5) on box GPU 2 ran at 66 s per record (about 6 hours
# for 321 records; the A6000's FP64 eigh is about 5 times slower than the H100's); it moves to pod 1 lane 4 as engine_replay_v7. Kills the GPU
# runner and its python child only when their command lines still match; logs to engine/status.txt.
cd ${QI2_ROOT}/engine || exit 1
stop_one() {
  local pid=$1 want=$2 args
  args=$(ps -o args= -p "$pid" 2>/dev/null)
  if [ -z "$args" ]; then echo "pid $pid gone"; return; fi
  case "$args" in *"$want"*) kill "$pid" && echo "killed $pid ($want)";; *) echo "pid $pid kept: $args";; esac
}
stop_one 2321438 "launch_engine_gpu_v4.sh 2 code/list_gpu2_v14.txt"
stop_one 2325929 "engine_replay_v5.py --run H100b/C3prod24/c24_rs64_A_R2_tau8_ell8_avg"
echo "GPU2 replay5_c24_rs64_ell8 stopped (moved to pod 1 lane 4 as engine_replay_v7; 66 s per record here) $(date -u +%H:%M:%S) UTC" >> status.txt
