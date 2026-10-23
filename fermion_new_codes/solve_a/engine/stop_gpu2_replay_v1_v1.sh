#!/bin/bash
# Stops box GPU 2's engine_replay_v1 run on the r_s 32 tau 8 set member (superseded by engine_replay_v2, which the pod already ran on the
# same record; the box rerun on v2 is the cross-hardware check): exact PID and argument checks, one process, no pattern kills. Builder analyst.
PID=$1
args=$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)
case "$args" in
  *code/engine_replay_v1.py*--run\ H100/C3set/set17_rs32_A_R2_tau8_ell4_avg*) ;;
  *) echo "PID $PID is not the GPU 2 replay v1 run: $args"; exit 1;;
esac
[ "$(stat -c %U /proc/$PID)" = "jon" ] || { echo "not jon's"; exit 1; }
kill -TERM $PID && echo "stopped $PID: $args"
