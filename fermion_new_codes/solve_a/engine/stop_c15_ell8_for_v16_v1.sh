#!/bin/bash
# Spec 18.1's restart rule (the Weyl-tested verification took 26.1 percent of the v15 ell 8 control's record time, above 20 percent): the
# v15 ell 8 control on GPU 0 and its runner stop, so that the control restarts on engine_collapse_v16 (nsub = 1 first; the trajectory
# version 15's bitwise, checked at r_s 16 in C3eq/compare_v16_v15_rs16.md). Exact PIDs, args checked. Builder analyst.
stop() {
  local P=$1; local WANT=$2
  local A=$(ps -o args= -p $P 2>/dev/null)
  if [ "$A" = "$WANT" ]; then kill $P && echo "stopped $P"; else echo "PID $P args do not match (got: '$A'); nothing stopped"; fi
}
stop 693008 "bash code/launch_engine_gpu_v4.sh 0 code/list_gpu0_v7.txt none none"
stop 696053 "${QI2_ROOT}/python code/engine_collapse_v15.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 8 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --out C3 --gpu 0 --tag c15_rs64_A_R2_tau8_ell8"
