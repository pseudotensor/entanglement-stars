#!/bin/bash
# stop_inbox3_runs_v1.sh (builder analyst): inbox #3 of 2026-10-16 (the averaging control is a fixed-preparation ladder; re-solved-preparation
# ell runs are a record, not the control; the (0.1, 1/16) bracket is optional and runs only on a GPU no critical item needs). Stops this lane's
# r_s 32 ell 8 control on box GPU 2 (the builder's interim choice) and the (0.1, 1/16) CPU bracket chains; each PID is killed only when its
# command line still matches; the launchers first, then their python children.
cd ${QI2_ROOT}/engine || exit 1
stop_one() {
  local pid=$1 want=$2 args
  args=$(ps -o args= -p "$pid" 2>/dev/null)
  if [ -z "$args" ]; then echo "pid $pid gone"; return; fi
  case "$args" in *"$want"*) kill "$pid" && echo "killed $pid ($want)";; *) echo "pid $pid kept: $args";; esac
}
stop_one 3082169 "launch_engine_gpu_v4.sh 2 code/list_gpu2_v15.txt"
for p in 2841002 2841008 2841009 2841011; do stop_one $p "launch_star_v7_brackets_v3.sh"; done
sleep 1
stop_one 3086501 "engine_collapse_v25.py --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --nsites 1024 --ell 8"
stop_one 2841016 "engine_star_v7.py --m 0.1 --g 0.0625 --ell 8 --nsites 4096"
stop_one 2841017 "engine_star_v7.py --m 0.1 --g 0.0625 --ell 16 --nsites 4096"
stop_one 2841018 "engine_star_v7.py --m 0.1 --g 0.0625 --ell 4 --nsites 4096"
echo "GPU2 c25_rs32_A_R2_tau8_ell8_avg stopped (inbox #3: re-solved-preparation ell runs are a record, not the control) $(date -u +%H:%M:%S) UTC" >> status.txt
echo "Sfam9 (0.1, 1/16) bracket stopped (inbox #3: optional, only on a GPU no critical item needs) $(date -u +%H:%M:%S) UTC" >> Sfam9/status.txt
