#!/bin/bash
# The lead's turn 27 (spec 19.2, 19.7), on the box: the v16 point-sampled ell 8 control on GPU 0 and its runner stop (the ell control runs
# averaged on the pod; GPU 0 takes the point-sampled r_s 64 control on v17); the set's box queues on GPUs 2 and 1 stop before starting (the
# set stays on the pod; GPU 2 takes vacflow v9, C5.3', C-capture; GPU 1 J6, J3', the ell 16 control); the J3' and C-capture waiters keyed on
# the v15 H100 primary stop (rekeyed on the production run). Exact PIDs, args checked. Builder analyst.
stop() {
  local P=$1; local WANT=$2
  local A=$(ps -o args= -p $P 2>/dev/null)
  if [ "$A" = "$WANT" ]; then kill $P && echo "stopped $P"; else echo "PID $P args do not match (got: '$A'); nothing stopped"; fi
}
stop 1466582 "bash code/launch_engine_gpu_v4.sh 0 code/list_gpu0_v8.txt none none"
stop 1469691 "${QI2_ROOT}/python code/engine_collapse_v16.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 8 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --out C3 --gpu 0 --tag c16_rs64_A_R2_tau8_ell8"
stop 1527556 "bash code/launch_engine_gpu_v4.sh 2 code/list_gpu2_v10.txt none none"
stop 1527557 "bash code/launch_engine_gpu_v4.sh 1 code/list_gpu1_v16.txt none none"
stop 1266272 "bash code/launch_obs_J3v3_primary_v10.sh"
stop 1266273 "bash code/launch_engine_capture_v2_v6.sh"
