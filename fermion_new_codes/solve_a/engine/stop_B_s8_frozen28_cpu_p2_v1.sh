#!/bin/bash
# stop_B_s8_frozen28_cpu_p2_v1.sh (builder analyst): stops B.6's T/r_s 28.0 frozen job on pod 2's CPU (launch_single_object_B_s8_frozen28_p2cpu_v4.sh,
# PID 1409883; its python PID 1409890), because the same run finished on pod 1's GPU (lane 2c, S4one/B_s8_frozen28_gpu.json, exit 0 at
# 22:07:28 UTC). Each PID is stopped after checking its command line: the launcher first, then the python job.
cd ${QI2_ROOT}/engine || exit 1
chk_kill() {
  pid=$1; want=$2
  cl=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$cl" in
    *"$want"*) kill -TERM "$pid" && echo "stopped $pid at $(date -u +%H:%M:%S) UTC";;
    *) echo "NOT stopped $pid: its command line '$cl' lacks '$want'";;
  esac
}
chk_kill 1409883 "bash code/launch_single_object_B_s8_frozen28_p2cpu_v4.sh"
sleep 1
chk_kill 1409890 "code/engine_single_object_B_v1.py --mode frozen --gpu -1 --record C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json --fields C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz --fields-module engine_fields_v16 --launch-T 28.0"
sleep 2
for p in 1409883 1409890; do kill -0 "$p" 2>/dev/null && echo "$p still alive" || echo "$p gone"; done
echo "B_s8_frozen28 (launcher v4, PID 1409883; python 1409890) stopped by code/stop_B_s8_frozen28_cpu_p2_v1.sh at $(date -u +%H:%M:%S) UTC: the same run finished on pod 1's GPU (lane 2c)" >> S4one/status_B.txt
