#!/bin/bash
# pod5_queue_v4.sh (the lane's analyst builder; engine spec 43.2 (2)): queue v3's recursion and item 3's variants again after their failed
# first starts (page_scatter_v5 and the v5 page modules were missing on pod 5; copied from the box), one after another beside the ell 2a member
# (CPU-bound, launched by queue v3 at 11:09 UTC): cut8, cut16, v6jA_s4gen, v6jA_s4cub. Usage (pod 5): setsid nohup bash pod5_queue_v4.sh &
set -u
W=/workspace; LOG=$W/pod5_queue_v4.log; C=$W/engine/code
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
log "queue v4 start"
H=$W/s11v35; TAG=s11_ball_rs16_ell2_tau2_v35from20
cd $W/page || exit 2; mkdir -p runs/s11 logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for NAME in cut8 cut16; do
  case "$NAME" in cut8) CUT="--R-cut-rs 8 --R-collar-rs 5.333333333333333";; cut16) CUT="--R-cut-rs 16 --R-collar-rs 10.666666666666666";; esac
  log "start s11pagei_$NAME"; nice -n 10 python code_v6/page_history_v6i.py --record $H/collapse_$TAG.json --fields $H/collapse_${TAG}_fields.npz --fields-key applied --nsites 1024 --channels 1,2 $CUT --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --join-rs 32,64 --max-wall-hours 20 --device cuda:0 --threads 8 --out runs/s11 --tag s11pagei_$NAME > logs/s11pagei_$NAME.log 2>&1; log "exit s11pagei_$NAME $?"
done
cd $W/page; mkdir -p runs/v6j
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
HP="--record $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-key applied"
BASE="$HP --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --max-wall-hours 20 --device cuda:0"
log "start v6jA_s4gen"; nice -n 10 python code_v6/page_history_v6j.py $BASE --collector-select gen --out runs/v6j/v6jA_s4gen --tag v6jA_s4gen --resume > logs/v6jA_s4gen.log 2>&1; log "exit v6jA_s4gen $?"
log "start v6jA_s4cub"; nice -n 10 python code_v6/page_history_v6j.py $BASE --cubic-window 32,40 --out runs/v6j/v6jA_s4cub --tag v6jA_s4cub --resume > logs/v6jA_s4cub.log 2>&1; log "exit v6jA_s4cub $?"
log "queue v4 done"
