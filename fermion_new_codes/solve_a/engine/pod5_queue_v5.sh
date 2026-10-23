#!/bin/bash
# pod5_queue_v5.sh (the lane's analyst builder; engine spec 43.2 (2), 43.3 (2)): the held-field controls' reruns inserted ahead of item 3's
# second variant. Queue v4 started v6jA_s4gen at 11:22:23 UTC before the reruns were built; it is left to finish (PID given, its command line
# checked), queue v4's shell stopped by its PID so it does not start v6jA_s4cub. Then, one at a time beside the ell 2a member: v6kA_c15 and
# v6kA_c30 (page_history_v6k with --collar-split; every stored v6eA_c15/c30 setting reproduced, gen_hold_args_v1.py; their init states
# md5-checked against the box), then v6jA_s4cub (queue v4's line). Each GPU start waits for 44000 MiB free (a hold's device peak about 44.7 GB
# on pod 1); PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on the holds (allocation only); a hold that exits nonzero is resumed once
# (--resume, its saved state). Usage (pod 5): setsid nohup bash pod5_queue_v5.sh <s4gen PID> > /dev/null 2>&1 < /dev/null &
set -u
W=/workspace; LOG=$W/pod5_queue_v5.log
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
GEN=${1:?the v6jA_s4gen PID}
log "queue v5 start; v6jA_s4gen PID $GEN ($(tr '\0' ' ' < /proc/$GEN/cmdline 2>/dev/null | grep -o 'v6jA_s4gen' | head -1))"
while [ -e /proc/$GEN ] && tr '\0' ' ' < /proc/$GEN/cmdline 2>/dev/null | grep -q v6jA_s4gen; do sleep 60; done
log "v6jA_s4gen ended; its log's last line: $(tail -1 $W/page/logs/v6jA_s4gen.log | cut -c1-200)"
gate() { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${f:-0}" -ge 44000 ] && { log "gate: $f MiB free"; break; }; log "gate: $f MiB free < 44000, waiting"; sleep 120; done; }
cd $W/page; mkdir -p runs/v6k logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
declare -A MD5=([15]=6e965961e789b7365607ffb5029121ff [30]=8ed7981ecf4f4fb6c01d14785f30b04f)
for n in 15 30; do
  st=$W/page/holdstates/v6bA_s4_state_at_0$n.000.pt
  m=$(md5sum $st 2>/dev/null | cut -d' ' -f1)
  [ "$m" = "${MD5[$n]}" ] || { log "v6kA_c$n: init state md5 $m against ${MD5[$n]}: not launched"; continue; }
  mapfile -t ARGS < $W/page/hold_args_c$n.txt
  for attempt in 1 2; do
    gate
    log "start v6kA_c$n (attempt $attempt)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 python code_v6/page_history_v6k.py "${ARGS[@]}" >> logs/v6kA_c$n.log 2>&1
    rc=$?; log "exit v6kA_c$n $rc"
    [ $rc -eq 0 ] && break
  done
done
cd $W/page; mkdir -p runs/v6j
HP="--record $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-key applied"
BASE="$HP --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --max-wall-hours 20 --device cuda:0"
gate
log "start v6jA_s4cub"; nice -n 10 python code_v6/page_history_v6j.py $BASE --cubic-window 32,40 --out runs/v6j/v6jA_s4cub --tag v6jA_s4cub --resume > logs/v6jA_s4cub.log 2>&1; log "exit v6jA_s4cub $?"
log "queue v5 done"
