#!/bin/bash
# v36member_switch_pod1_v1.sh (the lane's analyst builder; engine spec 43.2 (1)): every 10 min, compare_member_floor_v1 of the v36 member
# against the v35 jvp member at twice the floor; FAIL on any common record: the v36 member stopped by its PID (the jvp member continues);
# PASS over the first 4 common records after T = 0 and the v36 member's clock at or past the jvp member's latest: the jvp member (PID given,
# its command line read first) stopped, its records kept. Usage (pod 1): setsid nohup bash v36member_switch_pod1_v1.sh <jvp PID> <v36 python PID> &
JPID=${1:?}; VPID=${2:?}; D=${QI2_ROOT}/v36member; LOG=$D/switch.log; C=${QI2_ROOT}/code
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
log "switch v1: jvp PID $JPID ($(tr '\0' ' ' < /proc/$JPID/cmdline | cut -c1-70)), v36 PID $VPID"
while [ -e /proc/$VPID ] && [ -e /proc/$JPID ]; do
  sleep 600
  python $C/compare_member_floor_v1.py --run $D/c32_rs64_A_R2_tau8_ell2_prep4_avg_v36 --ref ${QI2_ROOT}/c32_rs64_A_R2_tau8_ell2_prep4_avg_v35jvp --label-run v36 --label-ref v35jvp --md $D/gate_v36_vs_v35jvp.md --out-json $D/gate_v36_vs_v35jvp.json > /dev/null 2>&1 || continue
  r=$(python - <<'PY'
import json
d = json.load(open("${QI2_ROOT}/gate_v36_vs_v35jvp.json")); rows = [r for r in d["rows"] if r["T_over_rs"] > 0]
bad = [r for r in rows if not r.get("within_twice_floor")]
J = lambda p: [x for x in json.load(open(p))["records"] if x.get("E") is not None]
tv = J("${QI2_ROOT}/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_v36.json")[-1]["T"]; tj = J("${QI2_ROOT}/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_v35jvp.json")[-1]["T"]
print("FAIL" if bad else ("SWITCH" if len(rows) >= 4 and tv >= tj else "WAIT"), len(rows), len(bad), tv / 64, tj / 64)
PY
)
  log "$r"
  case "$r" in
    FAIL*) kill $VPID && log "v36 member stopped (PID $VPID): a common record outside twice the floor"; exit 1;;
    SWITCH*) if tr '\0' ' ' < /proc/$JPID/cmdline | grep -q "engine_collapse_v35.py.*v35jvp"; then kill $JPID && log "the v35 jvp member stopped by PID $JPID"; fi; exit 0;;
  esac
done
log "a run ended before the switch"
