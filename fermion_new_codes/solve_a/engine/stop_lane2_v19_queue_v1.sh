#!/bin/bash
# Engine spec 24.1 (1): "every new launch runs on v16": stops, by exact PID after checking the arguments, the H100's lane-2 launcher
# (launch_h100b_v19_v2.sh, PID 1018, whose remaining items would launch on v19) and the lane-2b runner (PID 3275, the ell 16 control on
# v19); the running c_LZ 2 member (PID 2912) continues; the items move to queue_lane2c on engine_collapse_v24. Builder analyst.
chk() { local P=$1 W=$2; local A=$(ps -o args= -p $P 2>/dev/null); case "$A" in *"$W"*) kill $P && echo "stopped $P ($A)";; *) echo "not stopped: PID $P has args '$A'";; esac; }
chk 1018 "bash code/launch_h100b_v19_v2.sh 832"
chk 3275 "bash code/pod_queue_runner_v1.sh lane2b queue_lane2b.txt 1018"
ps -o pid,args -p 2912 | tail -1 | cut -c1-120
