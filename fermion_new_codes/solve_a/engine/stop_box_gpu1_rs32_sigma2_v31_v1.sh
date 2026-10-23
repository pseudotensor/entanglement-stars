#!/bin/bash
# Stops the box GPU 1 copy of the r_s/a 32 pair's sigma/a 2 member on engine_collapse_v31 (Newton; PID 1899643): its pod 1 copy (lane 4,
# since 12:02 UTC, inbox #12) runs the same arguments about four times faster on the H100. Exact PID and argument checks; the sigma/a 4
# member (GPU 2) is not touched. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PID=1899643
ARGS=$(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)
case "$ARGS" in
  *"code/engine_collapse_v31.py"*"--rs 32"*"--ell 2 --mix newton"*"--tag c31_rs32_A_R2_tau2_ell2_prep4_avg_newton"*) ;;
  *) echo "PID $PID is not the GPU 1 r_s 32 sigma/a 2 member: '$ARGS'"; exit 2;;
esac
kill $PID; sleep 3
if kill -0 $PID 2>/dev/null; then echo "still alive"; exit 3; fi
echo "GPU1 c31_rs32_A_R2_tau2_ell2_prep4_avg_newton stopped (its pod 1 copy, lane 4, runs the same member on the H100; inbox #12) $(date -u +%H:%M:%S) UTC" >> status.txt
echo stopped
