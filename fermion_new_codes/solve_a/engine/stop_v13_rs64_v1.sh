#!/bin/bash
# Stop the r_s 64 runs on engine_collapse_v13 (started 19:44 UTC, at T/rs 0.62) so they restart on engine_collapse_v15, which records the
# interior fields' change per self-consistency iteration as the goal of 19:57 UTC requires (the trajectory is v13's: v14 and v15 reproduce
# it bitwise at r_s 16), with their runners and the two waiters keyed on the v13 primary. Exact PIDs, args checked. Builder analyst.
stop() {
  local P=$1; local WANT=$2
  local A=$(ps -o args= -p $P 2>/dev/null)
  if [ "$A" = "$WANT" ]; then kill $P && echo "stopped $P"; else echo "PID $P args do not match (got: '$A'); nothing stopped"; fi
}
stop 414259 "bash code/launch_engine_gpu_v4.sh 1 code/list_gpu1_v13.txt none none"
stop 414260 "bash code/launch_engine_gpu_v4.sh 0 code/list_gpu0_v6.txt none none"
stop 417411 "${QI2_ROOT}/python code/engine_collapse_v13.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --out C3 --gpu 0 --tag primary_rs64_A_R2_tau8_v13"
stop 417378 "${QI2_ROOT}/python code/engine_collapse_v13.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 8 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --out C3 --gpu 0 --tag c13_rs64_A_R2_tau8_ell8"
stop 414261 "bash code/launch_engine_capture_v2_v4.sh"
stop 414262 "bash code/launch_obs_J3v3_primary_v8.sh"
