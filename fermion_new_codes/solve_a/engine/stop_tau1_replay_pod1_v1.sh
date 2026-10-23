#!/bin/bash
# stop_tau1_replay_pod1_v1.sh (builder analyst): the r_s 32 tau 1 member's replay (engine_replay_v7, 1280 records) holds pod 1 lane 2 ahead
# of the massless relaunch (the order of work's item 3); the compute plan puts replays on the second H100, so it moves there. Kills the PID
# only when its command line still matches. Run on pod 1 from ${QI2_ROOT}/engine.
cd ${QI2_ROOT}/engine || exit 1
pid=21383; want="engine_replay_v7.py --run C3set25/c25_rs32_A_R2_tau1_ell4_avg"
args=$(ps -o args= -p "$pid" 2>/dev/null)
case "$args" in *"$want"*) kill "$pid" && echo "killed $pid ($want)";; *) echo "pid $pid kept or gone: $args";; esac
echo "replay7_c25_rs32_tau1 stopped on pod 1 to move to the second H100 $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
