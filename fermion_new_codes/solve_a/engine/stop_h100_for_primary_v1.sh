#!/bin/bash
# The owner's handoffs addendum (22:05 UTC, binding): "Keep the H100 for the primary first; give it the pod to itself if the other jobs there
# slow it". The two r_s 64-size jobs beside the production (the ell 8 averaged control in its initial data; the 2048-site r_s 32 run at T/rs
# 7) stop; both rerun after the production. The two light r_s 32 runs within about 10 minutes of their ends finish. Exact PIDs, args checked.
# Builder analyst (runs on the pod).
stop() {
  local P=$1; local WANT=$2
  local A=$(ps -o args= -p $P 2>/dev/null)
  if [ "$A" = "$WANT" ]; then kill $P && echo "stopped $P"; else echo "PID $P args do not match (got: '$A'); nothing stopped"; fi
}
stop 3812 "bash code/launch_h100_ell8_17_v1.sh"
stop 3816 "/usr/bin/python3.11 code/engine_collapse_v17.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 8 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --site-fields averaged --out C3prod --gpu 0 --tag c17_rs64_A_R2_tau8_ell8_avg"
stop 2331 "/usr/bin/python3.11 code/engine_collapse_v17.py --m 0.2 --g 0.25 --rs 32 --nsites 2048 --T-over-rs 40 --reading A --ell 4 --tau 8 --kappa-sea 10 --reference R2 --site-fields averaged --out C3set --gpu 0 --tag set17_rs32_A_R2_tau8_ell4_n2048_avg"
