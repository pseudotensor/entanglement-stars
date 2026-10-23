#!/bin/bash
# Engine spec 27.3 and 27.6 (lead turn 35): "the r_s 64 tau 2 member to 21.5 (688 records) on the first free lane after the tau 4 control
# passes 21, in either outcome", on engine_collapse_v25 (ops v3). Waits until the tau 4 control (lane 3, c19_rs64_tau4_T0) has printed a
# record at T/rs >= 21 or has exited, then runs the member from T = 0 with the production's settings at tau 2. Started by the lane 1 queue
# runner from ${QI2_ROOT}/engine. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
S=h100b_status.txt
until grep -q "c19_rs64_tau4_T0 exit" $S || tr '\r' '\n' < logs_h100b_c19_rs64_tau4_T0.log | grep -o -E "^T/rs [0-9]+\.[0-9]+" | awk '$2 + 0 >= 21.0 {f = 1} END {exit !f}'; do sleep 60; done
echo "tau2 r_s 64 guard passed (the tau 4 control at T/rs 21 or ended) $(date -u +%H:%M:%S) UTC" >> $S
exec python code/engine_collapse_v25.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 2 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 21.5 --out C3prod25 --gpu 0 --tag c25_rs64_A_R2_tau2_ell4_avg
