#!/bin/bash
# gate_v6s_32k_after_init.sh (busi lane, #309 item 3): waits for the 32k initial vacuum cache (the host init process gone and the file present), then
# runs the real-init 32k smoke on GPU $1: kappa 1, T 0 to 0.5 r_s sub-stepped, the 0.5 r_s record held to 1.5 r_s, readouts every 0.5 r_s (the same
# clocks and hold as the 16k gate F2, for the comparison), the impurity at every readout.
GPU=${1:-2}; W=${QI2_ROOT}/page_verify; S=${QI2_ROOT}/code_d13; CACHE=$W/cache/B2_n32768_k1_T0_Phiplus.pt
cd $W
while ps -eo args | grep -v grep | grep -q page_interior_init_v6s; do sleep 60; done
if [ ! -s $CACHE ]; then echo "I_32k_real REFUSED $(date -u +%FT%TZ): no cache $CACHE" >> status.txt; exit 1; fi
echo "init cache present $(date -u +%FT%TZ) md5 $(md5sum $CACHE | cut -c1-8)" >> status.txt
bash $S/gate_v6s_box.sh I_32k_real page_history_v6s.py $GPU 1-1 1.5 --nsites 32768 --lean-interior --interior-every-rs 0.5 --hold-fields-from 0.5 --hold-step-rs 1 --save-every 10 --impurity-every-rs 0 --threads 8 --init-cache $CACHE
