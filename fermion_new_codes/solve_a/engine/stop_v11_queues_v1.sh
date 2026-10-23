#!/bin/bash
# Stop the lane's engine_collapse_v11 queues and their waiters so the runs restart on engine_collapse_v12 (the owner's request of 15 October:
# the GPU jobs optimized for speed and memory; v12 reproduces v11's records within the outer tolerance, half the eigendecompositions).
# Runners first (so they start nothing after their child ends), then the two v11 runs, then the three waiters keyed on v11 names.
# Every PID is checked against its exact args. Builder analyst.
stop() {
  local P=$1; local WANT=$2
  local A=$(ps -o args= -p $P 2>/dev/null)
  if [ "$A" = "$WANT" ]; then kill $P && echo "stopped $P ($A)"; else echo "PID $P args do not match (got: '$A'); nothing stopped"; fi
}
stop 523861 "bash code/launch_engine_gpu_v4.sh 0 code/list_gpu0_v4.txt 2412342 none"
stop 453889 "bash code/launch_engine_gpu_v4.sh 2 code/list_gpu2_v7.txt 937061 none"
stop 259402 "${QI2_ROOT}/python code/engine_collapse_v11.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 8 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --out C3 --gpu 0 --tag c11_rs64_A_R2_tau8_ell8"
stop 4113041 "${QI2_ROOT}/python code/engine_collapse_v11.py --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --nsites 1024 --T-over-rs 40 --out C3 --reading A --ell 4 --tau 8 --kappa-sea 10 --reference R2 --gpu 0 --tag c11_rs32_A_R2_tau8_ell4"
stop 453891 "bash wait_then_J45final_ell16_v3.sh"
stop 3953233 "bash code/launch_engine_capture_v2_v2.sh"
stop 3953234 "bash code/launch_obs_J3v3_primary_v6.sh"
