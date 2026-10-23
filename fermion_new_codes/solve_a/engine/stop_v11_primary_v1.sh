#!/bin/bash
# Stop the v11 primary (PID 4012286, started 19:07:59 UTC by the runner just before that runner was stopped): the owner asked for the GPU
# jobs to be optimized for speed and memory before the long runs. Exact PID, args checked. Builder analyst.
P=4012286
A=$(ps -o args= -p $P 2>/dev/null)
case "$A" in
  "${QI2_ROOT}/python code/engine_collapse_v11.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --T-over-rs 40 --kappa-sea 10 --reference R2 --out C3 --gpu 0 --tag primary_rs64_A_R2_tau8_v11") kill $P && echo "stopped $P";;
  *) echo "PID $P args do not match (got: '$A'); nothing stopped";;
esac
