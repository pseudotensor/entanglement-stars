#!/bin/bash
# Stops box GPU 1's engine_replay_v3 run on the production (moved to the H100, where it takes about 40 minutes against about 3 hours here):
# exact PID and argument checks, one process, no pattern kills. Builder analyst.
PID=$1
args=$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)
case "$args" in
  *code/engine_replay_v3.py*--run\ H100b/C3prod19/c19_rs64_A_R2_tau8_ell4_avg_r11*) ;;
  *) echo "PID $PID is not the box replay v3 run: $args"; exit 1;;
esac
[ "$(stat -c %U /proc/$PID)" = "jon" ] || { echo "not jon's"; exit 1; }
kill -TERM $PID && echo "stopped $PID: $args"
