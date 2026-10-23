#!/bin/bash
# Stop the two r_s 64 runs on engine_collapse_v12 (still near their start) so they restart on engine_collapse_v13 (M1 sparse, the same
# trajectory bitwise: results/engine/C3/compare_v13_v12_rs16.md), with their runners and the two waiters keyed on the v12 primary.
# The GPU 2 runner and its r_s 32 run are kept. Every PID is checked against its exact args. Builder analyst.
stop() {
  local P=$1; local WANT=$2
  local A=$(ps -o args= -p $P 2>/dev/null)
  if [ "$A" = "$WANT" ]; then kill $P && echo "stopped $P"; else echo "PID $P args do not match (got: '$A'); nothing stopped"; fi
}
stop 300503 "bash code/launch_engine_gpu_v4.sh 1 code/list_gpu1_v12.txt none none"
stop 300504 "bash code/launch_engine_gpu_v4.sh 0 code/list_gpu0_v5.txt none none"
stop 303932 "${QI2_ROOT}/python code/engine_collapse_v12.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --out C3 --gpu 0 --tag primary_rs64_A_R2_tau8_v12"
stop 303898 "${QI2_ROOT}/python code/engine_collapse_v12.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 8 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --out C3 --gpu 0 --tag c12_rs64_A_R2_tau8_ell8"
stop 300506 "bash code/launch_engine_capture_v2_v3.sh"
stop 300507 "bash code/launch_obs_J3v3_primary_v7.sh"
