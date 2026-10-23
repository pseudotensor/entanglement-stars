#!/bin/bash
# stop_replay_v9_p2lane2_v2.sh (builder analyst): stops this builder's orphaned engine_replay_v9 job on the second H100 (PID 35731, set17 tau 16,
# started by launch_replay_v9_p2lane2_v1.sh at 20:13:42 UTC just before that launcher was stopped), after checking its command line; it would
# fail at its first trapped record (the at-R_t scalar conversion).
P=35731
cp=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null)
case "$cp" in *engine_replay_v9.py*set17_rs32_A_R2_tau16_ell4_avg*) kill $P && echo "job $P sent SIGTERM $(date -u +%H:%M:%S) UTC";; *) echo "job $P not running or differs: '$cp'";; esac
for i in $(seq 1 40); do kill -0 $P 2>/dev/null || break; sleep 0.25; done
kill -0 $P 2>/dev/null && echo "job $P still alive" || echo "job $P gone $(date -u +%H:%M:%S) UTC"
