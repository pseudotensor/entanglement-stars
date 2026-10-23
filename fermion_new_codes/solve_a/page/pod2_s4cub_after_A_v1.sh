#!/bin/bash
# pod2_s4cub_after_A_v1.sh (the lane's analyst builder; engine spec 44.3 (3); #54's permission): on pod 2, after A to 100 (page_history_v6i,
# PID given, its command line holding s4to100) ends, item 3's cubic sub-step variant v6jA_s4cub (page_history_v6j --cubic-window 32,40 from T = 0,
# A's settings to 40, queue v4's line) beside the v32 sigma/a 2 member, the member untouched. The member's wall per record (its json's
# 'seconds') printed before the launch (the last 4 records) and every 30 min after it, and at the variant's end. Nothing else is placed here.
# Usage (pod 2): setsid nohup bash pod2_s4cub_after_A_v1.sh <A PID> > /dev/null 2>&1 < /dev/null &
set -u
APID=${1:?}; W=/workspace; LOG=$W/pod2_s4cub_v1.log
MJ=$W/engine/C3lad32r2b/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b.json
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
walls() { python -c "
import json;r=[x for x in json.load(open('$MJ'))['records'] if x.get('E') is not None]
print('the member at T/r_s %.4f; its last 4 records %s s' % (r[-1]['T']/64, ', '.join('%.0f' % x.get('seconds', float('nan')) for x in r[-4:])))" 2>&1; }
log "start; A PID $APID ($(tr '\0' ' ' < /proc/$APID/cmdline 2>/dev/null | grep -o 's4to100' | head -1))"
while [ -e /proc/$APID ] && tr '\0' ' ' < /proc/$APID/cmdline 2>/dev/null | grep -q s4to100; do sleep 120; done
log "A ended; $(walls)"
cd $W/page; mkdir -p runs/v6j logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
HP="--record $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-key applied"
BASE="$HP --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --max-wall-hours 20 --device cuda:0"
log "start v6jA_s4cub"
nice -n 10 python code_v6/page_history_v6j.py $BASE --cubic-window 32,40 --out runs/v6j/v6jA_s4cub --tag v6jA_s4cub --resume > logs/v6jA_s4cub.log 2>&1 &
VP=$!
while [ -e /proc/$VP ]; do sleep 1800; [ -e /proc/$VP ] && log "beside v6jA_s4cub: $(walls)"; done
wait $VP; log "exit v6jA_s4cub $?; $(walls)"
