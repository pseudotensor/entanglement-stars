#!/bin/bash
# The lead's turn 27 (spec 19.2, 19.4, 19.7), on the pod: the v15 primary stops (the v17 averaged run is the production; its 56 records to
# T/rs 7 stand as the point-sampling comparison) and vf8 stops (superseded by engine_vacflow_v9's sharp and smoothed cuts on box GPU 2,
# "never on the pod"). Exact PIDs, args checked. Builder analyst.
stop() {
  local P=$1; local WANT=$2
  local A=$(ps -o args= -p $P 2>/dev/null)
  if [ "$A" = "$WANT" ]; then kill $P && echo "stopped $P"; else echo "PID $P args do not match (got: '$A'); nothing stopped"; fi
}
stop 602 "python code/engine_collapse_v15.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --out C3 --gpu 0 --tag primary_rs64_A_R2_tau8_v15_h100"
stop 1757 "/usr/bin/python3.11 code/engine_vacflow_v8.py --rs 32 --profile exact --m 0 --kmax 16 --tr 32 --hold 2 --every 0.5 --chain-current full --gpu 0 --out VF8 --tag vf8_rs32_exact_m0_full_R2"
