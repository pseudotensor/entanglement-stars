#!/bin/bash
# The pod after the production's rejection at T/rs 12.75 (22:32 UTC): (1) the production's state replayed exactly over its 102 records
# (engine_replay_v1: the exactness table against the stored shell totals; the occupied orbitals at T/rs 5, 10 and 12 in complex128 under
# states/, for the handoff's collapse_states folder); (2) beside it, the r_s 32 tau 4 run of the tau ladder on engine_collapse_v17 (tau 8
# and 16 ran on v17), reading A, R2, ell 4, kernel averaged. Usage (on the pod, detached): cd ${QI2_ROOT}/engine; nohup bash
# code/launch_h100_after_prod_v1.sh > logs_h100_after_prod_launch.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p states/c17_rs64_A_R2_tau8_ell4_avg REPLAY C3set
( echo "replay start $(date -u +%H:%M:%S) UTC"
  OMP_NUM_THREADS=4 /usr/bin/python3.11 code/engine_replay_v1.py --run C3prod/c17_rs64_A_R2_tau8_ell4_avg --gpu 0 --slices 5,10,12 --dtype complex128 \
    --save-dir states/c17_rs64_A_R2_tau8_ell4_avg --md REPLAY/replay_c17_rs64_v1.md --out-json REPLAY/replay_c17_rs64_v1.json > logs_h100_replay_c17_rs64_v1.log 2>&1 < /dev/null
  echo "replay exit $? $(date -u +%H:%M:%S) UTC" ) &
( echo "set17_rs32_A_R2_tau4_ell4_avg start $(date -u +%H:%M:%S) UTC"
  OMP_NUM_THREADS=2 /usr/bin/python3.11 code/engine_collapse_v17.py --m 0.2 --g 0.25 --rs 32 --nsites 1024 --T-over-rs 40 --reading A --ell 4 --tau 4 --kappa-sea 10 \
    --reference R2 --site-fields averaged --out C3set --gpu 0 --tag set17_rs32_A_R2_tau4_ell4_avg > logs_h100_set17_rs32_A_R2_tau4_ell4_avg.log 2>&1 < /dev/null
  echo "set17_rs32_A_R2_tau4_ell4_avg exit $? $(date -u +%H:%M:%S) UTC" ) &
wait
echo "DONE $(date -u +%H:%M:%S) UTC"
