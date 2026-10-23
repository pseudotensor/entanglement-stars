#!/bin/bash
# pod2_queue_after_A_v2.sh (the lane's analyst builder; engine spec 46.2 (1), (3)): replaces pod2_s4cub_after_A_v1.sh. On pod 2, after A to 100
# (page_history_v6i, PID given, its command line holding s4to100) ends, one job at a time beside the v32 sigma/a 2 member (never touched): the
# held-field reruns v6kA_c15 then v6kA_c30 (page_history_v6k --collar-split; the stored v6eA settings, hold_args_c15/c30.txt; init states
# md5-checked), then v6jA_s4cub (v1's line). C's sigma/a 2 member (the busi lane's; never touched here) takes the first opening after its v3 gate
# has passed: each job starts only at an opening (no device process on the card but the member's) that has lasted 15 min, so a C launch by the
# busi lane in that window takes it, and the queue waits while any other process holds the card. The guard (46.2 (1)): every 60 s the card's free
# memory; below 8192 MiB the running job (the builder's own) is stopped by its PID (command line read first) and re-queued at the next opening
# with 53000 MiB free (a hold's peak about 44.7 GB plus 8 GB), resumed from its saved state (--resume); a job exiting nonzero by itself is resumed
# once. The member's walls (its json's 'seconds', last 4 records) printed before and after each job and every 30 min during it.
# Usage (pod 2): setsid nohup bash pod2_queue_after_A_v2.sh <A PID> <the member's nvidia-smi pid> > /dev/null 2>&1 < /dev/null &
set -u
APID=${1:?}; MNP=${2:?}; W=/workspace; LOG=$W/pod2_queue_v2.log
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
log "queue v2 start; A PID $APID ($(tr '\0' ' ' < /proc/$APID/cmdline 2>/dev/null | grep -o 's4to100' | head -1)); the member's nvidia-smi pid $MNP"
while [ -e /proc/$APID ] && tr '\0' ' ' < /proc/$APID/cmdline 2>/dev/null | grep -q s4to100; do sleep 120; done
log "A ended; $(walls)"
cd $W/page; mkdir -p runs/v6k runs/v6j logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
declare -A MD5=([15]=6e965961e789b7365607ffb5029121ff [30]=8ed7981ecf4f4fb6c01d14785f30b04f)
HP="--record $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json --fields $W/engine/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-key applied"
CUB="$HP --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --max-wall-hours 20 --device cuda:0 --cubic-window 32,40 --out runs/v6j/v6jA_s4cub --tag v6jA_s4cub --resume"
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
for n in 15 30; do
  st=$W/page/holdstates/v6bA_s4_state_at_0$n.000.pt; m=$(md5sum $st 2>/dev/null | cut -d' ' -f1)
  [ "$m" = "${MD5[$n]}" ] || { log "v6kA_c$n: init state md5 $m against ${MD5[$n]}: not launched"; continue; }
  mapfile -t ARGS < $W/page/hold_args_c$n.txt
  run_job v6kA_c$n "page_history_v6k.py.*v6kA_c$n" python code_v6/page_history_v6k.py "${ARGS[@]}"
done
run_job v6jA_s4cub "page_history_v6j.py.*v6jA_s4cub" python code_v6/page_history_v6j.py $CUB
log "queue v2 done; $(walls)"
