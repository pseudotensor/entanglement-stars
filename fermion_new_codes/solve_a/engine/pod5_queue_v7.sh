#!/bin/bash
# pod5_queue_v7.sh (the lane's analyst builder; engine spec 46.2 (1), (2), (4)): queue v6 re-armed without the holds (they move to pod 2 after
# A, 46.2 (3)). Nothing of the builder's runs on pod 5 beside the ell 2a member now; C's sigma/a 4 member (the busi lane's, its own guard
# ${QI2_ROOT}/pod5_guard_C_v1.sh) is never touched here. Every 10 min: every process's device share, the card's free memory, the ell 2a
# member's clock (its json) and its share's growth per r_s since the last line; a REPORT line below 8192 MiB free (C's stop is its own guard's).
# When the ell 2a member ends: a line that the ell 4a member's slot is open (the builder's launch, 46.2 (4), under 43.2 (3)'s load rule), then
# the queue ends. Usage (pod 5): setsid nohup bash pod5_queue_v7.sh <ell 2a python PID> <its nvidia-smi pid> > /dev/null 2>&1 < /dev/null &
set -u
MP=${1:?}; NP=${2:?the member pid as nvidia-smi prints it}; LOG=${QI2_ROOT}/pod5_queue_v7.log; J=${QI2_ROOT}/collapse_pair_rs32_ell2a_v36.json
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
log "queue v7 start; the ell 2a member PID $MP ($(tr '\0' ' ' < /proc/$MP/cmdline 2>/dev/null | grep -o 'engine_collapse_v36.py' | head -1))"
PM=""; PT=""
while [ -e /proc/$MP ]; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  SH=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')
  M=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$NP '$1==p{print $2}')
  T=$(python3 -c "
import json
try:
    r = [x for x in json.load(open('$J'))['records'] if x.get('E') is not None]; print('%.4f' % (r[-1]['T'] / 32))
except Exception: print('')" 2>/dev/null)
  G="-"; if [ -n "$PM" ] && [ -n "$M" ] && [ -n "$T" ] && [ -n "$PT" ]; then G=$(python3 -c "d=$T-$PT; print('%+.0f MiB per r_s' % (($M-$PM)/d) if d > 0 else '-')"); fi
  log "free ${FREE} MiB; shares: $SH the ell 2a member ${M:-?} MiB (nvidia-smi pid $NP) at T/r_s ${T:-?}; growth $G"
  [ "${FREE:-99999}" -lt 8192 ] && log "REPORT: free ${FREE} MiB below 8192 (C's own guard stops C; the ell 2a member never stopped)"
  [ -n "$M" ] && PM=$M; [ -n "$T" ] && PT=$T
  sleep 600
done
log "the ell 2a member ended; the ell 4a member's slot is open (the builder's launch, 46.2 (4)); queue v7 done"
