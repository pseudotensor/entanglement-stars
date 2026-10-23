#!/bin/bash
# Engine spec 27.1 and 27.6 (lead turn 35): the tau 1 member of the r_s 32 ladder on engine_collapse_v25, on lane 2 after the r_s 64 ell 16
# control and before the 2048-site control; the tau 2 member's settings (c24_rs32_tau2: lane 2c, 00:41:41 to 02:27:07 UTC) with tau 1, so
# the wall time is about twice its 1 h 45 min. Started by the lane 2 queue runner from ${QI2_ROOT}/engine. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
exec python code/engine_collapse_v25.py --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --nsites 1024 --ell 4 --kappa-sea 10 --reference R2 --site-fields averaged --reading A --tau 1 --T-over-rs 40 --out C3set25 --gpu 0 --tag c25_rs32_A_R2_tau1_ell4_avg
