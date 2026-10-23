#!/bin/bash
# arm_ell4a_pod5_v1.sh (solve builder 6, analyst; engine spec 47.4 (1), (2), 47.6 (1)): on pod 5, waits for the ell 2a r_s/a 32 member (its python PID
# as $1, command line checked to hold pair_rs32_ell2a_v36) to end, then, under the load rule (no other compute process on the card and at least
# 50000 MiB free for 5 min), launches the ell 4a member: engine_collapse_v36b.py --pair-workers 2 with the ell 2a member's arguments at --ell 4,
# --slices-c128 every 1 r_s (the list form 1 to 40), its own dir ${QI2_ROOT}/pair_ell4a (states/ beside; the box's pair_mirror_ell4a_v1.sh
# moves every slice but the newest 2 to ${QI2_ROOT}/pair_ell4a_slices). Label: the v36b gate record results/engine/V36b/ (bitwise
# against v36 with the option off and on). Every record: the v36b entry's serial-fallback count printed; REPORT on the first nonzero count (nothing
# stopped). The guard: card free memory below 8192 MiB stops nothing of this member (a member is never stopped) and is logged as REPORT.
# Usage (pod 5): setsid nohup bash arm_ell4a_pod5_v1.sh <ell 2a python PID> > /dev/null 2>&1 < /dev/null &
set -u
P2=${1:?}; W=${QI2_ROOT}/engine; D=$W/pair_ell4a; LOG=${QI2_ROOT}/arm_ell4a_v1.log; TAG=pair_rs32_ell4a_v36b
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $LOG; }
log "armed pid $$; ell 2a PID $P2 ($(tr '\0' ' ' < /proc/$P2/cmdline 2>/dev/null | grep -o 'pair_rs32_ell2a_v36' | head -1))"
while [ -e /proc/$P2 ] && tr '\0' ' ' < /proc/$P2/cmdline 2>/dev/null | grep -q pair_rs32_ell2a_v36; do sleep 120; done
log "ell 2a member ended"
since=""
while true; do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$n" -eq 0 ] && [ "${f:-0}" -ge 50000 ]; then [ -z "$since" ] && since=$(date +%s); [ $(( $(date +%s) - since )) -ge 300 ] && break; else since=""; fi
  sleep 30
done
log "load rule met: $f MiB free, no compute process for 5 min"
mkdir -p $D/states; cd $D
SL=$(seq -s, 1 40)
python ${QI2_ROOT}/engine_collapse_v36b.py --m 0 --g 0.25 --rs 32 --nsites 2048 --ell 4 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 \
  --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 \
  --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto \
  --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0 --nk-jvp exact \
  --field-kernel-cache --v36-fixes --ball-orbitals 1:12 --T-over-rs 40 --slices-c128 $SL --slice-dir states --out . --tag $TAG --pair-workers 2 > $D/$TAG.log 2>&1 &
PY=$!
log "launched $TAG python PID $PY (engine_collapse_v36b --pair-workers 2; gate record results/engine/V36b/)"
seen=0; rep=0
while kill -0 $PY 2>/dev/null; do
  sleep 120
  line=$(python - <<PYX 2>/dev/null
import json
try:
    d = json.load(open("$D/collapse_$TAG.json")); r = [x for x in d["records"] if x.get("E") is not None]
    v = r[-1].get("v36b") or {}
    print(len(r), "%.4f" % (r[-1]["T"] / 32.0), v.get("serial_fallbacks"), "%.0f" % (r[-1].get("seconds") or 0))
except Exception as e:
    print("0 - - -")
PYX
)
  set -- $line; nrec=${1:-0}
  if [ "$nrec" != "$seen" ] && [ "$nrec" != 0 ]; then
    seen=$nrec; log "record $nrec T/r_s $2 serial_fallbacks $3 seconds $4"
    if [ "$rep" -eq 0 ] && [ "$3" != "None" ] && [ "$3" != "-" ] && [ "$3" != "0" ]; then rep=1; log "REPORT: first nonzero serial-fallback count $3 at record $nrec (nothing stopped)"; fi
  fi
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "${f:-99999}" -lt 8192 ] && log "REPORT: card free $f MiB below 8192 (the member is not stopped)"
done
wait $PY; log "$TAG exit $?"
