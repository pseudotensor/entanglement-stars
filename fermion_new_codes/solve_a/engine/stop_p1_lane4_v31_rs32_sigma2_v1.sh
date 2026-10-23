#!/bin/bash
# Stop the r_s/a 32 pair's sigma/a 2 v31 copy on pod 1 lane 4 (engine spec 33.5 (3): uncertified from T/rs 0.125 by 30.3; its records the
# v31 record): the exact PID and its arguments checked first; a status line in h100b_status.txt. Builder analyst.
PID=36565
cd ${QI2_ROOT}/engine || exit 1
ARGS=$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)
case "$ARGS" in
  *"code/engine_collapse_v31.py"*"--rs 32 "*"--ell 2 --mix newton"*"--tag c31_rs32_A_R2_tau2_ell2_prep4_avg_newton"*) ;;
  *) echo "PID $PID does not run the v31 copy: [$ARGS]"; exit 2;;
esac
kill -TERM $PID && echo "# lad32_sigma2_newton_p1 (PID $PID, engine_collapse_v31, the r_s/a 32 sigma/a 2 Newton copy) stopped (engine spec 33.5 (3): uncertified from T/rs 0.125 by 30.3; its records the v31 record) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
sleep 8; if ps -p $PID > /dev/null; then echo "PID $PID still alive"; else echo "PID $PID stopped"; fi
