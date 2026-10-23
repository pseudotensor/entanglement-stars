#!/bin/bash
# The H100 is shared by 7 runs (GPU at 100 percent; the r_s 64 production's eigendecompositions about 6 times slower than alone): the four
# r_s 32 lane loops stop starting new runs (each lane's running engine_collapse_v17 process continues to its end, reparented); the
# remaining set runs move to the box GPUs (list_gpu2_v10.txt, list_gpu1_v16.txt). Each loop shell is identified by its exact args and by its
# running child's tag. Builder analyst (runs on the pod).
stop_loop() {
  local P=$1; local CHILD=$2; local TAG=$3
  local A=$(ps -o args= -p $P 2>/dev/null); local C=$(ps -o args= -p $CHILD 2>/dev/null); local PP=$(ps -o ppid= -p $CHILD 2>/dev/null | tr -d ' ')
  if [ "$A" = "bash code/launch_h100_set17_v1.sh code/list_h100_set17_v1.txt" ] && [ "$PP" = "$P" ] && echo "$C" | grep -q -- "--tag $TAG\$"; then
    kill $P && echo "stopped loop $P (its run $CHILD $TAG continues)"
  else
    echo "PID $P / child $CHILD do not match (args '$A', child parent '$PP', child '$C'); nothing stopped"
  fi
}
stop_loop 2034 2258 set17_rs32_A_R2_tau8_ell4_avg
stop_loop 2035 2046 set17_rs32_A_R2_tau16_ell4_avg
stop_loop 2041 2327 set17_rs32_B_R2_tau8_ell4_R0x4_avg
stop_loop 2040 2331 set17_rs32_A_R2_tau8_ell4_n2048_avg
