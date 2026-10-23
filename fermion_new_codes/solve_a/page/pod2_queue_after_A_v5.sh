#!/bin/bash
# pod2_queue_after_A_v5.sh (solve builder 7, analyst; engine spec 48.5 (2), 48.2 (2), 48.3 (a)): version 4 (stopped by its PID before its first
# start) with --col-diag on the v6m run. Version 4 replaced pod2_queue_after_A_v3.sh (stopped by
# its PID with the velocity-admitted v6jA_s4cub_gen, its records kept). One at a time beside the v32 sigma/a 2 member (untouched), each at an
# opening (no other process on the card for 15 min and 53000 MiB free) under v3's guard (free memory below 8192 MiB stops the queued process by
# its PID, re-queued with its own --resume), the member's walls printed before, during and after. Order:
#   1. v6lA_s4gensplit: page_history_v6l on A's production line, --collector-select gen --collector-admit split (the corrected collector);
#   2. v6mA_s4gensplit: page_history_v6m (check (viii)), the same line, started only when code_v6/page_history_v6m.py and its gate marker
#      code_v6/V6M_GATE_PASS exist (the queue waits for them, its slot kept);
#   3. v6lA_s4cub_gensplit: v3's cubic window line on page_history_v6l with --collector-admit split;
#   4. v6lA2_far48_gensplit: the far collector (48.3 (a)): A2's line on B2's records (synced to ${QI2_ROOT}/B2 by the box's
#      sync_B2_to_pod2_v1.sh) on page_history_v6l with --R-cut-rs 48, gen and split, --follow from T = 0 to --T-end-rs 80.
# Usage (pod 2): setsid nohup bash pod2_queue_after_A_v5.sh <the member's nvidia-smi pid> > /dev/null 2>&1 < /dev/null &
set -u
MNP=${1:?}; W=/workspace; LOG=$W/pod2_queue_v5.log
MJ=$W/engine/C3lad32r2b/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b.json
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
walls() { python -c "
import json;r=[x for x in json.load(open('$MJ'))['records'] if x.get('E') is not None]
print('the member at T/r_s %.4f; its last 4 records %s s' % (r[-1]['T']/64, ', '.join('%.0f' % x.get('seconds', float('nan')) for x in r[-4:])))" 2>&1; }
others() { nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk -v m=$MNP '$1!=m' | wc -l; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
shares() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';'; }
opening() { local since=""; while true; do
  if [ "$(others)" -eq 0 ] && [ "$(free_mib)" -ge 53000 ]; then [ -z "$since" ] && since=$(date +%s)
    [ $(( $(date +%s) - since )) -ge 900 ] && { log "opening: $(free_mib) MiB free, no other process for 15 min"; return; }
  else since=""; fi; sleep 60; done; }
log "queue v5 start; the member's nvidia-smi pid $MNP; $(walls)"
cd $W/page; mkdir -p runs/v6l runs/v6m logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
HP="--record $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-key applied"
LINE="$HP --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 1 --max-wall-hours 20 --device cuda:0 --collector-select gen --collector-admit split --resume"
B2J=$W/engine/B2/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json; B2F=$W/engine/B2/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz
FAR="--record $B2J --fields $B2F --fields-key applied --nsites 8192 --channels 1,2 --R-cut-rs 48 --R-collar-rs 16 --join-rs 32,64 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 1 --device cuda:0 --resume --T-end-rs 80 --ledger-anchor-rs 12.125 --save-at-rs 80 --max-wall-hours 60 --record-ext $B2J --fields-ext $B2F --ext-after-rs 2 --follow --follow-poll-s 60 --follow-timeout-h 6 --follow-done-marker $W/engine/B2/EXT_DONE_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.txt --collector-select gen --collector-admit split"
run_job() {  # $1 tag, $2 pattern, rest the command
  local tag=$1 pat=$2; shift 2; local selfexit=0
  while true; do
    opening; log "start $tag; $(walls)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 "$@" >> logs/$tag.log 2>&1 &
    local P=$! G=0 n=0
    log "$tag pid $P"
    while [ -e /proc/$P ]; do
      sleep 60; n=$((n + 1)); f=$(free_mib)
      if [ "${f:-99999}" -lt 8192 ] && tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null | grep -q "$pat"; then
        kill $P; G=1; log "GUARD: free $f MiB below 8192: $tag stopped by PID $P; shares: $(shares)"; fi
      [ $((n % 30)) -eq 0 ] && [ -e /proc/$P ] && log "beside $tag: $(walls); free $f MiB; shares: $(shares)"
    done
    wait $P; rc=$?; log "exit $tag $rc; $(walls)"
    [ $G -eq 1 ] && { log "$tag re-queued (guard)"; continue; }
    [ $rc -eq 0 ] && return
    [ $selfexit -ge 1 ] && { log "$tag exited nonzero twice: left for the lead"; return; }
    selfexit=1; log "$tag resumed once"
  done
}
run_job v6lA_s4gensplit "page_history_v6l.py.*v6lA_s4gensplit" python code_v6/page_history_v6l.py $LINE --out runs/v6l/v6lA_s4gensplit --tag v6lA_s4gensplit
until [ -e code_v6/page_history_v6m.py ] && [ -e code_v6/V6M_GATE_PASS ]; do sleep 300; done
log "page_history_v6m and its gate marker present: $(cat code_v6/V6M_GATE_PASS | head -1)"
run_job v6mA_s4gensplit "page_history_v6m.py.*v6mA_s4gensplit" python code_v6/page_history_v6m.py $LINE --col-diag --out runs/v6m/v6mA_s4gensplit --tag v6mA_s4gensplit
run_job v6lA_s4cub_gensplit "page_history_v6l.py.*v6lA_s4cub_gensplit" python code_v6/page_history_v6l.py $LINE --cubic-window 32,40 --out runs/v6l/v6lA_s4cub_gensplit --tag v6lA_s4cub_gensplit
run_job v6lA2_far48_gensplit "page_history_v6l.py.*v6lA2_far48_gensplit" python code_v6/page_history_v6l.py $FAR --out runs/v6l/v6lA2_far48_gensplit --tag v6lA2_far48_gensplit
log "queue v5 done; $(walls)"
