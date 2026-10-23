#!/bin/bash
# stop_B_s8_launcher_v2_p2_v1.sh (builder analyst): stops only the bash launcher launch_single_object_B_s8_p2cpu_v2.sh (PID 284078) by its exact
# PID after checking its command line, so that it does not start its frozen run after the history run; its python history job (PID 284093) is
# left running and finishes on its own (its exit line will be missing from S4one/status_B.txt; its output S4one/B_s8_history_v2.json is
# written by the job itself). The frozen run starts from launch_single_object_B_s8_frozen_p2cpu_v3.sh in a separate call.
cd ${QI2_ROOT}/engine || exit 1
P=284078
cl=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null)
case "$cl" in
  *"bash code/launch_single_object_B_s8_p2cpu_v2.sh"*) kill -TERM $P && echo "stopped $P at $(date -u +%H:%M:%S) UTC";;
  *) echo "NOT stopped $P: its command line '$cl' differs";;
esac
sleep 2
kill -0 $P 2>/dev/null && echo "$P still alive" || echo "$P gone"
c=$(tr '\0' ' ' < /proc/284093/cmdline 2>/dev/null); case "$c" in *"--mode history"*) echo "history job 284093 still running";; *) echo "history job 284093 not running: '$c'";; esac
echo "the B.6 launcher v2 (PID 284078) stopped by code/stop_B_s8_launcher_v2_p2_v1.sh at $(date -u +%H:%M:%S) UTC; its history job (PID 284093) continues; the frozen run moves to launcher v3" >> S4one/status_B.txt
