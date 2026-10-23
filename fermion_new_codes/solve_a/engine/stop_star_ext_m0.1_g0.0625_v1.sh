#!/bin/bash
# Engine spec 27.4 (lead turn 35): "No ladder beyond N_B,max at (0.1, 1/64) or at (0.1, 1/16)". Stops the running (0.1, 1/16) extension
# (engine_star_v6 --m 0.1 --g 0.0625 ... --tag m0.1_g0.0625_ell8_ext, started by launch_star_v6_ext_v1.sh after its family ended at 02:54
# UTC); its converged member at N_B 8877.009 is kept in Sfam6/star_m0.1_g0.0625_ell8_ext.json. Only the exact PID with the exact arguments is
# signalled; the (0.2, 1/16) extension's bisection is untouched. Run on the box. Builder analyst.
PID=$1
cd ${QI2_ROOT}/engine || exit 1
args=$(ps -o args= -p $PID 2>/dev/null)
case "$args" in
  *"code/engine_star_v6.py --m 0.1 --g 0.0625 "*"--tag m0.1_g0.0625_ell8_ext"*) ;;
  *) echo "PID $PID is not the (0.1, 1/16) extension: '$args'; nothing done"; exit 1;;
esac
kill -TERM $PID && echo "star_v6 m0.1_g0.0625_ell8_ext stopped (engine spec 27.4: no ladder at (0.1, 1/16)) $(date -u +%H:%M:%S) UTC" >> status.txt
sleep 5; ps -o pid,args -p $PID >/dev/null 2>&1 && echo "still alive" || echo "PID $PID ended"
