#!/bin/bash
# gate_v6s_chain2.sh (busi lane, inbox #309, item 3): E2 the lean mode with the in-place propagator on the production chain (kappa 1, T 0 to 2 r_s,
# readouts every 2 r_s; against gate C's production readout); F the lean mode with the hold on 16384 sites (T 0 to 0.5 r_s sub-stepped, the record at
# 0.5 r_s held to 2.5 r_s in 1 r_s Chebyshev steps, readouts every 1 r_s, the contour cached on the held clocks).
GPU=${1:-2}
W=${QI2_ROOT}/page_verify; S=${QI2_ROOT}/code_d13
cd $W
bash $S/gate_v6s_box.sh E2_lean page_history_v6s.py $GPU 1-1 2 --lean-interior --interior-every-rs 2
bash $S/gate_v6s_box.sh F_hold16k page_history_v6s.py $GPU 1-1 2.5 --nsites 16384 --lean-interior --interior-every-rs 1 --hold-fields-from 0.5 --hold-step-rs 1 --save-every 1
echo "chain2 done $(date -u +%FT%TZ)" >> status.txt
