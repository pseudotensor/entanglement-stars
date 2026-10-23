#!/bin/bash
# v36member_switch_pod1_v3.sh (the lane's analyst builder; engine spec 45.3 (2)): the tightened v36 relaunch's monitor. Every 10 min:
# compare_member_floor_v2 of the tightened run (${QI2_ROOT}/v36member_r2) against the stopped v36 run's records (${QI2_ROOT}/v36member) at the floor
# itself (the test, 45.1 (2)), and against the stopped jvp copy (${QI2_ROOT}/jvpmember, to T/r_s 1.875) and the pod 2 v32 member (${QI2_ROOT}/v32ref,
# synced by the box's sync_v32_to_pod1_v1.sh) at twice the floor as a record; the latest common record's differences and their record-to-record
# growth, the fixes' counts and the device memory of every process printed. REPORT lines (nothing stopped): |dE/E0| above 1e-06 against the v32
# member; the run's device share above 36 GB; a record that meets the stored stop rule (fields 1e-06, |dE/E| 1e-07) but not the tightened one
# (verdict (iii): its kept residual is the member's attainable floor there). The run is stopped by its PID (command line read first) only on a
# mask inequality against the v32 member or on a record that fails to converge (41.3 (4): the engine's converged flag false with the kept
# residual or the confirming |dE/E| above the stored stop rule). Nothing else stops. Ends when the run ends.
# Usage (pod 1): setsid nohup bash v36member_switch_pod1_v3.sh <python PID> > /dev/null 2>&1 &
VPID=${1:?}; D=${QI2_ROOT}/v36member_r2; LOG=$D/switch_v3.log; C=${QI2_ROOT}/code
T=c32_rs64_A_R2_tau8_ell2_prep4_avg_v36tol10
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
log "switch v3: run PID $VPID ($(tr '\0' ' ' < /proc/$VPID/cmdline | grep -o 'engine_collapse_v36.py\|--stop-fields [0-9e.-]*\|--stop-E [0-9e.-]*' | tr '\n' ' '))"
while [ -e /proc/$VPID ]; do
  sleep 600
  MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')
  VM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$VPID '$1==p{print $2}')
  log "device memory (pid, MiB): $MEM run ${VM:-?} MiB"
  [ -n "$VM" ] && [ "$VM" -gt 34332 ] && log "REPORT: the run's device share ${VM} MiB is above 36 GB; nothing stopped"
  STOP=""
  for REF in v36 jvp v32; do
    case $REF in
      v36) R=${QI2_ROOT}/c32_rs64_A_R2_tau8_ell2_prep4_avg_v36; L=v36stopped; CR=floor;;
      jvp) R=${QI2_ROOT}/c32_rs64_A_R2_tau8_ell2_prep4_avg_v35jvp; L=v35jvp; CR=twice;;
      v32) R=${QI2_ROOT}/c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b; L=v32; CR=twice;;
    esac
    [ -e $(dirname $R)/collapse_$(basename $R).json ] || { log "$REF: no records"; continue; }
    [ -e $D/collapse_$T.json ] || { log "$REF: the run has no json yet"; continue; }
    python $C/compare_member_floor_v2.py --run $D/$T --ref $R --label-run v36tol10 --label-ref $L --criterion $CR --md $D/gate_v36tol10_vs_$L.md --out-json $D/gate_v36tol10_vs_$L.json > /dev/null 2>&1 || { log "$REF: compare error"; continue; }
    OUT=$(python - "$D/gate_v36tol10_vs_$L.json" "$REF" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); ref = sys.argv[2]; rows = d["rows"]
if not rows:
    print("none"); sys.exit()
x = rows[-1]; bad = [r for r in rows if not r["within"]]
g = lambda v, f: (f % v) if v is not None else "-"
s = x["solve_run"]
msg = "%s(%s): %d common to T/r_s %.4f, %d outside; last dE/E0 %+.2e (growth %s, increment %s) d min %+.2e (growth %s, increment %s) masks %s; run kept residual %s confirming dE %s converged %s; gap_none %s cache hits %s" % (
    ref, d["criterion"], len(rows), x["T_over_rs"], len(bad), x["dE"], g(x["dE_growth"], "%.2f"), g(x["dE_increment"], "%+.2e"), x["dmin"],
    g(x["dmin_growth"], "%.2f"), g(x["dmin_increment"], "%+.2e"), "equal" if x["masks_all_equal"] else "NOT EQUAL", g(s["kept_residual"], "%.2e"),
    g(s["confirming_dE"], "%.2e"), s["converged"], s["gap_none_substeps"], s["cache_hits"])
flags = []
if ref == "v32":
    if any(abs(r["dE"]) > 1e-6 for r in rows): flags.append("REPORT_dE")
    if any(not r["masks_all_equal"] for r in rows): flags.append("STOP_MASK")
print(msg + ((" FLAGS " + " ".join(flags)) if flags else ""))
PY
)
    log "$OUT"
    case "$OUT" in *REPORT_dE*) log "REPORT: |dE/E0| above 1e-06 against the v32 member (45.3 (2)); nothing stopped";; esac
    case "$OUT" in *STOP_MASK*) STOP="$STOP mask-$REF";; esac
  done
  # every record of the run itself (not only the common ones): the stop rule at the stored and the tightened tolerances
  OWN=$(python - "$D/collapse_$T.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("unreadable"); sys.exit()
flags = []; rs = float(d["args"]["rs"]); n = 0
for r in d["records"]:
    if r.get("E") is None: continue
    n += 1; s = r.get("solve_v32") or {}; kr = s.get("kept_residual"); cd = s.get("confirming_dE")
    if r.get("converged") is False:
        if (kr is None or kr >= 1e-6) or (cd is None or cd >= 1e-7): flags.append("STOP_UNCONV@%.4f" % (r["T"] / rs))
        else: flags.append("STALL@%.4f(kept %.2e, dE %.2e)" % (r["T"] / rs, kr, cd))
last = [r for r in d["records"] if r.get("E") is not None]
print("own: %d records to T/r_s %.4f" % (n, last[-1]["T"] / rs if last else -1) + ((" FLAGS " + " ".join(flags)) if flags else ""))
PY
)
  log "$OWN"
  case "$OWN" in *STALL@*) log "REPORT: a record meets the stored stop rule but not the tightened one (verdict (iii)); nothing stopped";; esac
  case "$OWN" in *STOP_UNCONV*) STOP="$STOP unconverged";; esac
  if [ -n "$STOP" ]; then
    if tr '\0' ' ' < /proc/$VPID/cmdline | grep -q "engine_collapse_v36.py.*v36tol10"; then kill $VPID && log "STOP: the run stopped by PID $VPID (a mask inequality against the v32 member or an unconverged record:$STOP)"; fi
    exit 1
  fi
done
log "the run ended"
