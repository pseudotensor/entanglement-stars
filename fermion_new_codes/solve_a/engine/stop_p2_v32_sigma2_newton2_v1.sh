#!/bin/bash
# Stop the r_s/a 64 sigma/a 2 v32 member with Krylov restarts 2 on the second H100 (engine spec 33.5 (1): its records the record of the
# restarts' cost; relaunched from T = 0 with --nk-restarts 0 in a separate call): the exact PID and its arguments checked first. Builder analyst.
PID=16989
cd ${QI2_ROOT}/engine || exit 1
ARGS=$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)
case "$ARGS" in
  *"code/engine_collapse_v32.py"*"--rs 64 "*"--nk-restarts 2 "*"--tag c32_rs64_A_R2_tau8_ell2_prep4_avg_newton2"*) ;;
  *) echo "PID $PID does not run the v32 member: [$ARGS]"; exit 2;;
esac
kill -TERM $PID && echo "# lad32_sigma2_newton2 (PID $PID, engine_collapse_v32 with --nk-restarts 2) stopped (engine spec 33.5 (1): the restarts cost 2.3 to 2.7 times v31's evaluations for the same step; its records the record of that cost) $(date -u +%H:%M:%S) UTC" >> h100b_status.txt
sleep 8; if ps -p $PID > /dev/null; then echo "PID $PID still alive"; else echo "PID $PID stopped"; fi
