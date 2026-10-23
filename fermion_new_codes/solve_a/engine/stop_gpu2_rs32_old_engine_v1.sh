#!/bin/bash
# The goal of 19:57 UTC (the lead's review, binding): the r_s = 32 set runs on the economy engine, built and validated first, and GPU 2
# takes the vacuum-flow r_s = 32 run with the R2 column. Stop the GPU 2 runner of list_gpu2_v8.txt, its running first r_s 32 run (v12, the
# old engine) and the J45 waiter keyed on that list's end. Exact PIDs, args checked. Builder analyst.
stop() {
  local P=$1; local WANT=$2
  local A=$(ps -o args= -p $P 2>/dev/null)
  if [ "$A" = "$WANT" ]; then kill $P && echo "stopped $P"; else echo "PID $P args do not match (got: '$A'); nothing stopped"; fi
}
stop 300505 "bash code/launch_engine_gpu_v4.sh 2 code/list_gpu2_v8.txt none none"
stop 303899 "${QI2_ROOT}/python code/engine_collapse_v12.py --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --nsites 1024 --T-over-rs 40 --out C3 --reading A --ell 4 --tau 8 --kappa-sea 10 --reference R2 --gpu 0 --tag c12_rs32_A_R2_tau8_ell4"
stop 300508 "bash wait_then_J45final_ell16_v4.sh"
