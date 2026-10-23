#!/bin/bash
# Stops the builder's own balls.py dense control at R 6, L 128 (|x| <= 6 convention; the review's control is |x| < 6, run instead as
# --radius 5.99 by launch_obs_ball_cut_ctrl_v2.sh).  Kills only PID $1 and only if its command line is exactly that run's.
PID=$1
WANT="${QI2_ROOT}/python ${QI2_ROOT}/balls.py measure --radius 6 --box 128 --method dense --mass 0 --center 0 0 0 --gpu -1 --output ctrl/balls_dense_R6_L128.json"
GOT=$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null | sed 's/ $//')
if [ "$GOT" == "$WANT" ]; then
  kill $PID && echo "stopped $PID" && echo "ctrl R6_L128 stopped by the builder $(date -u +%Y-%m-%dT%H:%M:%SZ) (|x| <= 6 convention; replaced by --radius 5.99)" >> ${QI2_ROOT}/status_ctrl_v1.txt
else
  echo "not stopped: PID $PID command line is [$GOT]"
fi
