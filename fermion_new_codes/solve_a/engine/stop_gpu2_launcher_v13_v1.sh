#!/bin/bash
# Stops box GPU 2's queue runner for list_gpu2_v13 (its next line, the (0.05, 1/4) extension with the old ell-control rule, is replaced by
# engine spec 26.4's per-member ladder); the runner's current child (the (0.1, 1/64) member, engine_star_v8) keeps running ("the current
# member finishes"): exact PID and argument checks, the bash runner only. Builder analyst.
PID=$1
args=$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)
case "$args" in
  "bash code/launch_engine_gpu_v4.sh 2 code/list_gpu2_v13.txt none none ") ;;
  *) echo "PID $PID is not the GPU 2 v13 runner: $args"; exit 1;;
esac
[ "$(stat -c %U /proc/$PID)" = "jon" ] || { echo "not jon's"; exit 1; }
kill -TERM $PID && echo "stopped $PID: $args"
