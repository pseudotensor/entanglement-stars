#!/bin/bash
# Stop the r_s/a 32 pair's sigma/a 2 v32 member with Krylov restarts 2 on box GPU 0 (engine spec 33.5 (2): "the same unless its end line
# at 0.125 is already written"; no end line: it converged at T/rs 0.125, kept residual 1.71e-08, and its records stay as the record), to be
# relaunched from T = 0 with --nk-restarts 0 in a separate call: the exact PID and its arguments checked first. Builder analyst.
PID=2925507
cd ${QI2_ROOT}/engine || exit 1
ARGS=$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)
case "$ARGS" in
  *"code/engine_collapse_v32.py"*"--rs 32 "*"--nk-restarts 2 "*"c32_rs32_A_R2_tau2_ell2_prep4_avg_newton2"*) ;;
  *) echo "PID $PID does not run the v32 r_s 32 member: [$ARGS]"; exit 2;;
esac
kill -TERM $PID && echo "GPU0 c32_rs32_A_R2_tau2_ell2_prep4_avg_newton2 stopped (engine spec 33.5 (2): no end line at T/rs 0.125, converged there; relaunched with --nk-restarts 0) $(date -u +%H:%M:%S) UTC" >> status.txt
sleep 8; if ps -p $PID > /dev/null; then echo "PID $PID still alive"; else echo "PID $PID stopped"; fi
