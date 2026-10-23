#!/bin/bash
# pod2_queue_after_A_v3.sh (solve builder 6, analyst; engine spec 47.7 (1), 47.6 (1); busi inbox #80): replaces pod2_queue_after_A_v2.sh (stopped
# by its PID; the c30 hold it started, python PID given as $1, is adopted, not restarted). Order: c15 (done, exit 0 18:32:31 UTC), c30 (running,
# watched here under the same guard: free memory below 8192 MiB stops it by its PID, then re-queued with its own --resume at the next opening),
# then the cubic window run under the corrected selector, v6jA_s4cub_gen (v1's s4cub line plus --collector-select gen, --save-every 1, its own
# tag and out); s4cub without gen is dropped (47.7 (1)); C's sigma/a 2 member runs on pod 3 (busi #80), so no slot is held for it here.
# Usage (pod 2): setsid nohup bash pod2_queue_after_A_v3.sh <c30 python PID> <the member's nvidia-smi pid> > /dev/null 2>&1 < /dev/null &
set -u
C30PID=${1:?}; MNP=${2:?}; W=/workspace; LOG=$W/pod2_queue_v3.log
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
log "queue v3 start; c30 PID $C30PID ($(tr '\0' ' ' < /proc/$C30PID/cmdline 2>/dev/null | grep -o 'v6kA_c30' | head -1)); the member's nvidia-smi pid $MNP"
cd $W/page; mkdir -p runs/v6k runs/v6j logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
declare -A MD5=([15]=6e965961e789b7365607ffb5029121ff [30]=8ed7981ecf4f4fb6c01d14785f30b04f)
HP="--record $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-key applied"
CUB="$HP --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 1 --max-wall-hours 20 --device cuda:0 --cubic-window 32,40 --collector-select gen --out runs/v6j/v6jA_s4cub_gen --tag v6jA_s4cub_gen --resume"
run_job() {  # $1 tag, $2 pattern, rest the command
  local tag=$1 pat=$2; shift 2; local selfexit=0
  while true; do
    opening; log "start $tag; $(walls)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 "$@" >> logs/$tag.log 2>&1 &
    local P=$! G=0 n=0
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
# c30: adopted while its python runs (the guard as run_job's); re-queued by run_job (its own --resume) only if the guard stopped it or it exited nonzero
G=0
while [ -e /proc/$C30PID ] && tr '\0' ' ' < /proc/$C30PID/cmdline 2>/dev/null | grep -q v6kA_c30; do
  sleep 60; f=$(free_mib)
  if [ "${f:-99999}" -lt 8192 ]; then kill $C30PID; G=1; log "GUARD: free $f MiB below 8192: v6kA_c30 stopped by PID $C30PID; shares: $(shares)"; break; fi
done
sleep 5; ex=$(grep -c '"final": true' runs/v6k/v6kA_c30/v6kA_c30_summary.json 2>/dev/null)
log "v6kA_c30 (adopted PID $C30PID) ended; guard $G; final flag count ${ex:-0}; $(walls)"
if [ $G -eq 1 ] || [ "${ex:-0}" -eq 0 ]; then
  mapfile -t ARGS < $W/page/hold_args_c30.txt
  run_job v6kA_c30 "page_history_v6k.py.*v6kA_c30" python code_v6/page_history_v6k.py "${ARGS[@]}"
fi
run_job v6jA_s4cub_gen "page_history_v6j.py.*v6jA_s4cub_gen" python code_v6/page_history_v6j.py $CUB
log "queue v3 done; $(walls)"
