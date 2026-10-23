#!/bin/bash
# stop_replay_v9_p2lane2_v1.sh (builder analyst): stops this builder's S5 launcher on the second H100 (launch_replay_v9_p2lane2_v1.sh, PID 34240)
# and its running engine_replay_v9 job (PID 35488, set17 tau 4), each after checking its command line: engine_replay_v9's at-R_t scalar
# conversion fails at the first trapped record (numpy's float() of a one-element array); the c17 parent's completed outputs are unaffected
# (that code path never ran there). The p2lane2 runner (PID 3090) is not touched.
L=34240; P=35488
cl=$(tr '\0' ' ' < /proc/$L/cmdline 2>/dev/null)
case "$cl" in *launch_replay_v9_p2lane2_v1.sh*) kill $L && echo "launcher $L sent SIGTERM $(date -u +%H:%M:%S) UTC";; *) echo "launcher $L not running or differs: '$cl'";; esac
cp=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null)
case "$cp" in *engine_replay_v9.py*set17_rs32_A_R2_tau4_ell4_avg*) kill $P && echo "job $P sent SIGTERM $(date -u +%H:%M:%S) UTC";; *) echo "job $P not running or differs: '$cp'";; esac
for i in $(seq 1 40); do kill -0 $P 2>/dev/null || break; sleep 0.25; done
kill -0 $P 2>/dev/null && echo "job $P still alive" || echo "job $P gone $(date -u +%H:%M:%S) UTC"
