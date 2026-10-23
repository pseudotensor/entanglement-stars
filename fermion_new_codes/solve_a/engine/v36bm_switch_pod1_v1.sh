#!/bin/bash
# v36bm_switch_pod1_v1.sh (solve builder 7, analyst; engine spec 48.1 (4)): v36member_switch_pod1_v4.sh's criterion unchanged, on the
# stored-tolerance member (engine_collapse_v36b --pair-workers 2, ${QI2_ROOT}/v36b_member, the tag in CURRENT_TAG). Every 10 min:
# compare_member_floor_v2 against the ended tightened run's 103 records (${QI2_ROOT}/v36member_r2, tag ..._v36tol10) at the floor and against the
# pod 2 v32 member (${QI2_ROOT}/v32ref, synced by the box's sync_v32_v3.sh) at twice the floor; the latest common record's differences, growth,
# the run's kept and confirming residuals, gap_none and cache counts, the device memory of every process printed. REPORT (nothing stopped):
# |dE/E0| above 1e-06 against the v32 member; the run's device share above 36 GB. The run is stopped by its PID (command line read first)
# only on a mask inequality against the v32 member or on a record that fails the stored stop rule (converged false with the kept residual
# above 1e-06 or the confirming |dE/E| above 1e-07). If the engine itself ends the run, its records and both residuals are the record and
# nothing is relaunched (46.1 (3)). Follows a guard restart (CURRENT_TAG). Usage (pod 1): setsid nohup bash v36bm_switch_pod1_v1.sh &
D=${QI2_ROOT}/v36b_member; LOG=$D/switch_v1.log; C=${QI2_ROOT}/code
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
mpid() { local t=$(cat $D/CURRENT_TAG 2>/dev/null); [ -n "$t" ] && ps -eo pid,args | grep "[e]ngine_collapse_v36b.py.*--tag $t\b" | grep -v "bash " | awk '{print $1}' | head -1; }
log "switch v1 start: tag $(cat $D/CURRENT_TAG) PID $(mpid)"
idle=0
while true; do
  sleep 600
  T=$(cat $D/CURRENT_TAG); VPID=$(mpid)
  if [ -z "$VPID" ]; then idle=$((idle + 1)); log "no member process (tag $T); last status: $(tail -1 $D/status.txt)"
    [ $idle -ge 3 ] && grep -q "v36bm exit.*tag $T" $D/status.txt && ! ps -eo args | grep -q "[g]uard_pod1_v36bm_v1.sh" && { log "the run ended: $(tail -1 $D/status.txt); records kept, nothing relaunched (46.1 (3))"; exit 0; }
    continue; fi
  idle=0
  MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')
  VM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$VPID '$1==p{print $2}')
  log "device memory (pid, MiB): $MEM run ${VM:-?} MiB"
  [ -n "$VM" ] && [ "$VM" -gt 36864 ] && log "REPORT: the run's device share ${VM} MiB is above 36 GB; nothing stopped"
  STOP=""
  for REF in tol10 v32; do
    case $REF in
      tol10) R=${QI2_ROOT}/c32_rs64_A_R2_tau8_ell2_prep4_avg_v36tol10; L=v36tol10; CR=floor;;
      v32) R=${QI2_ROOT}/c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b; L=v32; CR=twice;;
    esac
    [ -e $(dirname $R)/collapse_$(basename $R).json ] || { log "$REF: no records"; continue; }
    [ -e $D/collapse_$T.json ] || { log "$REF: the run has no json yet"; continue; }
    python $C/compare_member_floor_v2.py --run $D/$T --ref $R --label-run v36b --label-ref $L --criterion $CR --md $D/gate_v36b_vs_$L.md --out-json $D/gate_v36b_vs_$L.json > /dev/null 2>&1 || { log "$REF: compare error"; continue; }
    OUT=$(python - "$D/gate_v36b_vs_$L.json" "$REF" <<'PY'
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
    case "$OUT" in *REPORT_dE*) log "REPORT: |dE/E0| above 1e-06 against the v32 member; nothing stopped";; esac
    case "$OUT" in *STOP_MASK*) STOP="$STOP mask-$REF";; esac
  done
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
        if kr is None and cd is None: flags.append("UNREAD@%.4f(no kept residual stored)" % (r["T"] / rs))
        elif (kr is not None and kr > 1e-6) or (cd is not None and cd > 1e-7): flags.append("STOP_UNCONV@%.4f(kept %s, dE %s)" % (r["T"] / rs, kr, cd))
        else: flags.append("MET_STORED@%.4f(kept %.2e, confirming %s, dE %.2e, evaluations %s)" % (r["T"] / rs, kr, s.get("confirming_residual"), cd, s.get("evaluations")))
last = [r for r in d["records"] if r.get("E") is not None]
print("own: %d records to T/r_s %.4f" % (n, last[-1]["T"] / rs if last else -1) + ((" FLAGS " + " ".join(flags)) if flags else ""))
PY
)
  log "$OWN"
  case "$OWN" in *STOP_UNCONV*) STOP="$STOP unconverged";; esac
  if [ -n "$STOP" ]; then
    if tr '\0' ' ' < /proc/$VPID/cmdline | grep -q "engine_collapse_v36b.py.*--tag $T"; then
      echo "v36bm stopped by its switch tag $T ($STOP) $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $D/status.txt
      kill $VPID && log "STOP: the run stopped by PID $VPID (a mask inequality against the v32 member or an unconverged record:$STOP)"; fi
    exit 1
  fi
done
