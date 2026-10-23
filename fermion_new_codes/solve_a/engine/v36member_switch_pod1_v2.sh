#!/bin/bash
# v36member_switch_pod1_v2.sh (the lane's analyst builder; engine spec 44.1 (2), (4)): the v36 member's gate after the jvp member's stop. Every
# 10 min: compare_member_floor_v1 of the v36 member against the stopped v35 jvp member's stored records (${QI2_ROOT}/jvpmember) and against the pod 2
# v32 member's records (synced to ${QI2_ROOT}/v32ref by the box's sync_v32_to_pod1_v1.sh), both at twice the floor; prints the device memory of every
# process on the GPU (the v36 member and A2) at each comparison and a REPORT line when the v36 member's share passes 36 GB (nothing stopped).
# PASS over the first 4 common records after T = 0 against the jvp copy is logged once as the launch's confirmation; the comparison runs on
# through the stored copy's last record and against the v32 member as its records land. Any common record outside twice the floor in either
# comparison: the v36 member stopped by its PID (command line read first), launch_jvpmember_pod1_v2.sh started (the jvp member from T = 0),
# exit. Ends when the v36 member ends. Usage (pod 1): setsid nohup bash v36member_switch_pod1_v2.sh <v36 python PID> > /dev/null 2>&1 &
VPID=${1:?}; D=${QI2_ROOT}/v36member; LOG=$D/switch_v2.log; C=${QI2_ROOT}/code
T=c32_rs64_A_R2_tau8_ell2_prep4_avg_v36
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
log "switch v2: v36 PID $VPID ($(tr '\0' ' ' < /proc/$VPID/cmdline | grep -o 'engine_collapse_v36.py' | head -1))"
CONF=0
while [ -e /proc/$VPID ]; do
  sleep 600
  MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')
  VM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$VPID '$1==p{print $2}')
  log "device memory (pid, MiB): $MEM v36 ${VM:-?} MiB"
  [ -n "$VM" ] && [ "$VM" -gt 34332 ] && log "REPORT: the v36 member's device share ${VM} MiB is above 36 GB (44.1 (4)); nothing stopped"
  V=""
  for REF in jvp v32; do
    case $REF in jvp) R=${QI2_ROOT}/c32_rs64_A_R2_tau8_ell2_prep4_avg_v35jvp; L=v35jvp;; v32) R=${QI2_ROOT}/c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b; L=v32;; esac
    [ -e $R.json ] || [ -e $(dirname $R)/collapse_$(basename $R).json ] || { V="$V $REF:none"; continue; }
    python $C/compare_member_floor_v1.py --run $D/$T --ref $R --label-run v36 --label-ref $L --md $D/gate_v36_vs_$L.md --out-json $D/gate_v36_vs_$L.json > /dev/null 2>&1 || { V="$V $REF:error"; continue; }
    r=$(python -c "
import json
d = json.load(open('$D/gate_v36_vs_$L.json')); rows = [r for r in d['rows'] if r['T_over_rs'] > 0]
bad = [r for r in rows if not r.get('within_twice_floor')]
print(('FAIL' if bad else 'ok') + ':%d:%d:%s' % (len(rows), len(bad), ('%.4f' % rows[-1]['T_over_rs']) if rows else '-'))
")
    V="$V $REF:$r"
  done
  log "gate:$V"
  case "$V" in
    *FAIL*) if tr '\0' ' ' < /proc/$VPID/cmdline | grep -q "engine_collapse_v36.py.*_v36"; then kill $VPID && log "FAIL: the v36 member stopped by PID $VPID; the jvp member relaunched from T = 0 (launch_jvpmember_pod1_v2.sh)"; fi
            setsid nohup bash ${QI2_ROOT}/launch_jvpmember_pod1_v2.sh > /dev/null 2>&1 < /dev/null &
            exit 1;;
  esac
  n=$(echo "$V" | grep -o 'jvp:ok:[0-9]*' | cut -d: -f3)
  [ $CONF -eq 0 ] && [ -n "$n" ] && [ "$n" -ge 4 ] && { CONF=1; log "PASS: the first $n common records after T = 0 within twice the floor against the stopped jvp copy: the launch confirmed"; }
done
log "the v36 member ended"
