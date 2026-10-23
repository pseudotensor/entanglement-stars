#!/bin/bash
# pull_s8half_box_v2.sh (busi, inbox #461): v1 with the card name, the last-copy and hard-stop times as env (v1 fixed h8x, 20:36, 20:45). ON THE BOX, beside C_half450 on the card. Every 120 s: each new slice on the card whose size
# is stable for 60 s is copied to ${QI2_ROOT}/h8x<tag>/T<clock>/ md5-checked (card md5 = box md5, else retried), then the
# run files (json, fields, handoff, spectra npz, log.txt, RUN_LOG, smi) are refreshed md5-checked into ${QI2_ROOT}/run and
# into every slice dir whose clock the json now passes, and resume_check_v1.py marks it RESUMABLE (the json is written every 5 records).
# A slice verified on the box and no longer the newest is removed from the card (card disk). After the card's RC appears: one last pass, exit.
# HARD STOP 20:45 UTC: no new slice copy starts after 20:36 (a 16 GB copy took ~8 min h100-f -> box); the run files are refreshed to the end.
#   CARD="ssh ... USER@IP" ROOT=${QI2_ROOT}/busi_s8half nohup bash pull_s8half_box_v1.sh > ${QI2_ROOT}/pull.out 2>&1 &
set -u
CARD="${CARD:?the card ssh command} -n -o ConnectTimeout=20 -o BatchMode=yes"; ROOT=${ROOT:-${QI2_ROOT}/busi_s8half}
TH=lad_F256_s8_fbt05_m0p2_rs256_n4096_R04_k15_tau8_ell8_prep4_A_R2_avg_newton_v36nq_rd13mfb_half450
PY=${QI2_ROOT}/python; RC=${QI2_ROOT}/resume_check_v1.py
CN=${CN:-h8x}; B=${QI2_ROOT}/pull_$CN; MB=${QI2_ROOT}/$TH; mkdir -p $B/run $MB
LAST_SLICE_START=$(date -d ${LAST_COPY:-2026-10-22T20:36:00Z} +%s); HARD=$(date -d ${HARD_STOP:-2026-10-22T20:45:00Z} +%s)
log(){ echo "$(date -u +%FT%TZ) $*" >> $B/PULL_LOG.txt; }
pullf(){ # card file -> box file, md5-checked; 0 on success
  local a b; $CARD "cat $1" > $2.new 2>/dev/null; a=$($CARD "md5sum < $1" 2>/dev/null | cut -c1-32); b=$(md5sum < $2.new | cut -c1-32)
  [ -n "$a" ] && [ "$a" = "$b" ] && mv $2.new $2 && return 0; rm -f $2.new; return 1; }
runfiles(){ local ok=1
  for s in .json _fields.npz _handoff.npz _spectra.npz; do pullf $ROOT/run/collapse_${TH}$s $B/run/collapse_${TH}$s || ok=0; done
  for f in run/log.txt RUN_LOG.txt logs/smi.csv RC; do $CARD "test -e $ROOT/$f" && { pullf $ROOT/$f $B/run/$(basename $f) || ok=0; }; done
  return $((1-ok)); }
log "start (card root $ROOT)"; fin=0
while true; do
  now=$(date +%s); [ -n "$($CARD "cat $ROOT/RC 2>/dev/null")" ] && fin=1
  for f in $($CARD "ls $ROOT/states/ 2>/dev/null" | grep "^orbitals_${TH}_T.*\.npz$"); do
    c=$(echo $f | sed -E 's/.*_T([0-9.]+)\.npz$/\1/'); d=$MB/T$c; [ -e $d/.ok ] && continue
    [ $(date +%s) -ge $LAST_SLICE_START ] && { log "T$c not copied: past the last-copy time; it stays on the card"; continue; }
    s1=$($CARD "stat -c %s $ROOT/states/$f"); sleep 60; [ "$s1" = "$($CARD "stat -c %s $ROOT/states/$f")" ] || continue
    mkdir -p $d; t0=$(date +%s)
    if pullf $ROOT/states/$f $d/$f; then echo $f > $d/SLICE; touch $d/.copied; log "T$c copied md5 $(md5sum < $d/$f | cut -c1-32) ($(( $(date +%s) - t0 )) s)"; else log "T$c md5 mismatch, retry next pass"; fi
  done
  runfiles && log "run files refreshed (json last T/rs $($PY -c "import json;J=json.load(open('$B/run/collapse_$TH.json'));print(J['records'][-1]['T']/256)" 2>/dev/null))" || log "run files: a file was in write, retry"
  for d in $(ls -d $MB/T* 2>/dev/null); do
    [ -e $d/.copied ] && [ ! -e $d/.ok ] || continue; f=$(cat $d/SLICE)
    for s in .json _fields.npz _handoff.npz _spectra.npz; do cp -p $B/run/collapse_${TH}$s $d/; done
    (cd $d && $PY -I $RC slice $d $TH $f > CHECK.txt; grep '^RESUMABLE' CHECK.txt | awk '{print $2}' > RESUMABLE)
    if [ "$(cat $d/RESUMABLE)" = yes ]; then (cd $d && md5sum collapse_* orbitals_* > MD5SUMS && touch .ok); log "$(basename $d) RESUMABLE yes (.ok)"; fi
  done
  newest=$($CARD "ls $ROOT/states/ 2>/dev/null" | grep "^orbitals_${TH}_T" | sort | tail -1)
  for d in $(ls -d $MB/T* 2>/dev/null); do f=$(cat $d/SLICE 2>/dev/null); [ -e $d/.copied ] && [ -n "$f" ] && [ "$f" != "$newest" ] && $CARD "test -e $ROOT/states/$f" && { $CARD "rm -f $ROOT/states/$f"; log "$f removed from the card (on the box md5-checked)"; }; done
  [ $fin = 1 ] && { runfiles; log "card RC: $(cat $B/run/RC 2>/dev/null); last pass done; end"; exit 0; }
  [ $(date +%s) -ge $HARD ] && { log "HARD STOP reached"; exit 0; }
  sleep 120
done
