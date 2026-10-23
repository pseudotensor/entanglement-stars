#!/bin/bash
# stop_massless_relaunch_v1.sh (builder analyst): engine spec 30.2 and 30.6 (3): "the queued relaunch withdrawn ... (its line kept in the
# queue file as 'withdrawn, 30.2')"; the relaunch had moved from pod 1 lane 2 to box GPU 2 after its preparation failed on the pod. Kills the
# GPU runner and its python child only when their command lines still match.
cd ${QI2_ROOT}/engine || exit 1
stop_one() {
  local pid=$1 want=$2 args
  args=$(ps -o args= -p "$pid" 2>/dev/null)
  if [ -z "$args" ]; then echo "pid $pid gone"; return; fi
  case "$args" in *"$want"*) kill "$pid" && echo "killed $pid ($want)";; *) echo "pid $pid kept: $args";; esac
}
stop_one 308920 "launch_engine_gpu_v4.sh 2 code/list_gpu2_v16.txt"
stop_one 312531 "engine_collapse_v26.py --massless-shell --m 0 --g 0.25 --rs 32 --nsites 1024"
echo "GPU2 c26ms_rs32_k0.4_w20_tau2 stopped: the relaunch is withdrawn by engine spec 30.2 (replaced by the nsub test on engine_collapse_v28) $(date -u +%H:%M:%S) UTC" >> status.txt
