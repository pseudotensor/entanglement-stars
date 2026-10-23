#!/bin/bash
# busi_resume_audit_v1.sh (busi lane, lane_z 9ae0d5c4, inbox #293 OWNER ORDER, 2026-10-19). Runs on the box (compute-host) from cron, hourly.
# For every job in ${QI2_ROOT}/busi_jobs.conf:
#  (a) finds its newest resumable state (ladder/twofield: the newest complex128 orbitals slice at least 2 min old plus the run's json, fields,
#      handoff, spectra (and s11) npz; v6o: the newest <tag>_k<K>_state.pt with its summary/d13 files; q_H: the running member's channel checkpoint);
#  (b) copies it to ${QI2_ROOT}/resume_mirror<card>/<tag>/<state>/ (pods have no rsync: ssh streams, 4 parallel dd streams above 1 GB;
#      box-local jobs: rsync --link-dest), md5 checked against the source (run files: source md5 before and after the copy must agree);
#      keeps the newest 2 complete states per run; skips copying with an ALERT when /data has < 150 GB free;
#  (c) checks the mirrored set with resume_check_v1.py (the engine's own --resume-slice/--resume-run, --resume, or checkpoint checks);
#  (d) appends one line per job to ${QI2_ROOT}/busi.log and writes ${QI2_ROOT}/busi_ALERT.txt when any job has no mirrored
#      resumable state younger than 90 min of wall time (running jobs), cannot resume, stopped before its end, or its card is unreachable;
#      removes the ALERT file when all are fine.
# Never deletes on the pods (read-only there: ls, stat, dd, md5sum, ps, python reading npz). Uses the box's ${QI2_ROOT}/.ssh key. No pkill/pgrep.
set -u
export LC_ALL=C PATH=/usr/local/bin:/usr/bin:/bin
CONF=${1:-${QI2_ROOT}/busi_jobs.conf}
AUD=${QI2_ROOT}/resume_audit; MIR=${QI2_ROOT}/resume_mirror; LOG=$AUD/busi.log; ALERT=$AUD/busi_ALERT.txt
HERE=${QI2_ROOT}/resume_audit; BPY=${QI2_ROOT}/python; CHK=$HERE/resume_check_v1.py
MINFREE_GB=150; MAXAGE_MIN=90; KEEP=2; BIG=$((1024*1024*1024)); NSTREAM=4
SSHO=(-n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o LogLevel=ERROR)
mkdir -p $AUD $MIR
NOW=$(date -u +%s); NOWI=$(date -u +%FT%TZ)
ALERTS=(); RUNLOG=$AUD/busi_last_run.txt; : > $RUNLOG
say() { echo "$(date -u +%FT%TZ) $*" >> $RUNLOG; }

rsh() { local h=$1 p=$2; shift 2; timeout ${RT:-900} ssh "${SSHO[@]}" -p $p $h "$@" 2>>$RUNLOG; }

# the remote finder: newest complex128 orbitals slice (>= 120 s old, readable) in <dir> for <tag>: "SLICE <realpath> <size> <mtime> <T/rs from the name>"
FIND_PY='import numpy as np,glob,os,sys,time,re
d,tag=sys.argv[1],sys.argv[2]
fs=sorted(glob.glob(os.path.join(d,"orbitals_%s_T*.npz"%tag)),key=lambda f:os.stat(f).st_mtime,reverse=True)
now=time.time()
for f in fs:
    st=os.stat(f)
    if now-st.st_mtime<120: continue
    try:
        Z=np.load(f); dt=str(Z["dtype"])
    except Exception as e:
        print("BAD",f,repr(e)); continue
    if dt!="complex128": continue
    m=re.search(r"_T([0-9.]+)\.npz$",f)
    print("SLICE",os.path.realpath(f),st.st_size,int(st.st_mtime),m.group(1) if m else "?"); break
else:
    print("NONE")'
FIND_B64=$(printf '%s' "$FIND_PY" | base64 -w0)

free_gb() { df -BG --output=avail /data | tail -1 | tr -dc 0-9; }

# pull_remote host port remote_path local_path size [mutable]: copy and md5-check; returns 0 when md5 equal
pull_remote() {
  local h=$1 p=$2 r=$3 l=$4 sz=$5 mut=${6:-0} a m1 m2 ml
  for a in 1 2 3; do
    m1=""; [ $mut = 1 ] && m1=$(rsh $h $p "md5sum '$r'" | cut -c1-32)
    rm -f "$l.tmp" "$l".part*
    if [ "$sz" -gt $BIG ]; then
      local cb=$(( (sz / (16*1024*1024) + NSTREAM) / NSTREAM )) i pids=()
      for i in $(seq 0 $((NSTREAM-1))); do
        ( RT=3600 rsh $h $p "dd if='$r' bs=16M skip=$((i*cb)) count=$cb 2>/dev/null" > "$l.part$i" ) & pids+=($!)
      done
      wait "${pids[@]}"
      cat $(for i in $(seq 0 $((NSTREAM-1))); do echo "$l.part$i"; done) > "$l.tmp"; rm -f "$l".part*
    else
      RT=1800 rsh $h $p "cat '$r'" > "$l.tmp"
    fi
    m2=$(rsh $h $p "md5sum '$r'" | cut -c1-32); ml=$(md5sum "$l.tmp" | cut -c1-32)
    if [ -n "$m2" ] && [ "$ml" = "$m2" ] && { [ $mut = 0 ] || [ "$m1" = "$m2" ]; }; then
      mv "$l.tmp" "$l"; echo "$ml  $(basename $l)" >> "$(dirname $l)/MD5SUMS"; return 0
    fi
    say "md5 mismatch $r attempt $a (src $m1/$m2 local $ml)"
  done
  rm -f "$l.tmp"; return 1
}

# prune: keep the newest $KEEP complete states (.ok) of a run dir; drop stale partials (box only)
prune() {
  local base=$1 d n=0
  for d in $(ls -1dt $base/*/ 2>/dev/null); do
    d=${d%/}
    if [ -e $d/.ok ]; then n=$((n+1)); [ $n -gt $KEEP ] && { say "prune $d"; rm -rf -- "$d"; }
    elif [ $(( NOW - $(stat -c %Y $d) )) -gt 10800 ]; then say "prune stale partial $d"; rm -rf -- "$d"; fi
  done
}

# newest complete mirrored state of a run: echoes "<dir> <src_mtime> <resumable>"
best_state() {
  local base=$1 d
  for d in $(ls -1dt $base/*/ 2>/dev/null); do d=${d%/}; [ -e $d/.ok ] && { echo "$d $(cat $d/SRC_MTIME 2>/dev/null || echo 0) $(cat $d/RESUMABLE 2>/dev/null || echo no)"; return; }; done
  echo "- 0 no"
}

running_count() { # host port match
  local h=$1 p=$2 m=$3
  if [ $h = local ]; then ps -eo args= | awk -v m="$m" '$1 ~ /python[0-9.]*$/ && index($0, m) > 0' | wc -l
  else rsh $h $p "ps -eo args= | awk '\$1 ~ /python[0-9.]*\$/ && index(\$0, \"$m\") > 0' | wc -l"; fi
}

FREE=$(free_gb); LOWSPACE=0
if [ "${FREE:-0}" -lt $MINFREE_GB ]; then LOWSPACE=1; ALERTS+=("ALL: /data free ${FREE} GB < ${MINFREE_GB} GB: mirroring skipped this hour (audit only)"); fi
say "start free ${FREE}G conf $CONF"

while read -r JOB CARD KIND HOST PORT PY RUNDIR TAG SDIR PM EXPECT OPTS; do
  case "$JOB" in ''|\#*) continue;; esac
  OPTS=${OPTS:-}; SCLK=""; sd=/nonexistent; base=$MIR/$CARD/$TAG; mkdir -p $base
  run="?"; clock="-"; age="-"; mirr=no; md5="-"; resm=no; final=no; note=""
  # running?
  if [ "$HOST" != local ]; then
    if ! rsh $HOST $PORT true; then run="?"; note="card unreachable"
    else run=$( [ "$(running_count $HOST $PORT $PM)" -gt 0 ] && echo yes || echo no ); fi
  else run=$( [ "$(running_count local - $PM)" -gt 0 ] && echo yes || echo no ); fi

  case $KIND in
  ladder|twofield)
    if [ "$run" != "?" ]; then
      F=$(rsh $HOST $PORT "$PY -c \"\$(echo $FIND_B64 | base64 -d)\" '$SDIR' '$TAG'" | grep -E '^(SLICE|NONE)' | head -1)
      if [ "${F%% *}" = SLICE ]; then
        read -r _ SP SSZ SMT SCLK <<< "$F"; sname=$(basename $SP)
        sd=$base/T$SCLK
        if [ ! -e $sd/.ok ] && [ $LOWSPACE = 0 ]; then
          rm -rf -- $sd.partial; mkdir -p $sd.partial; ok=1
          pull_remote $HOST $PORT $SP $sd.partial/$sname $SSZ 0 || ok=0
          for suf in "" _fields _handoff _spectra _s11; do
            [ $ok = 1 ] || break
            ext=npz; [ -z "$suf" ] && ext=json
            rf=$RUNDIR/collapse_${TAG}${suf}.$ext
            rsz=$(rsh $HOST $PORT "stat -L -c %s '$rf' 2>/dev/null")
            if [ -z "$rsz" ]; then [ "$suf" = _s11 ] && continue; ok=0; say "$JOB missing $rf"; break; fi
            pull_remote $HOST $PORT $rf $sd.partial/collapse_${TAG}${suf}.$ext $rsz 1 || ok=0
          done
          if [ $ok = 1 ]; then
            echo $SMT > $sd.partial/SRC_MTIME; echo "$HOST -p $PORT $SP ; run $RUNDIR/$TAG" > $sd.partial/SOURCE
            rm -rf -- $sd; mv $sd.partial $sd; touch $sd/.ok; md5=ok
          else md5=bad; note="$note copy/md5 failed for T$SCLK;"; fi
        fi
        # (c) on the newest complete mirrored state
        if [ -e $sd/.ok ] && [ ! -e $sd/CHECK.txt ]; then
          C=$(nice -n 10 $BPY $CHK slice $sd $TAG $sname 2>&1); echo "$C" > $sd/CHECK.txt
          echo "$C" | awk '$1=="RESUMABLE"{print $2}' > $sd/RESUMABLE
          echo "--resume-slice $sd/$sname --resume-run $sd/$TAG" > $sd/RESUME_ARGS
        fi
        [ -e $sd/.ok ] && md5=ok
        final=$(awk '$1=="FINAL"{print $2}' $sd/CHECK.txt 2>/dev/null); final=${final:-no}
        [ "$(cat $sd/RESUMABLE 2>/dev/null)" = yes ] || note="$note $(grep '^WHY' $sd/CHECK.txt 2>/dev/null | cut -c5-);"
      elif [ "${F%% *}" = NONE ]; then note="$note no complex128 slice on the card;"
      else note="$note slice finder failed;"; fi
    fi ;;
  v6o)
    ST=$(ls -1t $RUNDIR/${TAG}_k*_state.pt 2>/dev/null | head -1)
    if [ -n "$ST" ]; then
      P=$(basename ${ST%_state.pt}); SMT=$(stat -c %Y $ST)
      SCLK=$(grep "state saved at record" $SDIR 2>/dev/null | tail -1 | sed -n 's/.*(T\/r_s \([0-9.]*\)).*/\1/p'); SCLK=${SCLK:-x}
      sd=$base/${P#${TAG}_}_T${SCLK}_$SMT
      if [ ! -e $sd/.ok ] && [ $LOWSPACE = 0 ]; then
        prev=$(best_state $base | cut -d' ' -f1); LD=(); [ "$prev" != - ] && LD=(--link-dest=$prev)
        rm -rf -- $sd.partial; mkdir -p $sd.partial; ok=1
        for f in $RUNDIR/${P}_state.pt $RUNDIR/${P}_summary.json $RUNDIR/${P}_d13.json $RUNDIR/${P}_d13.npz $RUNDIR/${TAG}_ksum.json; do
          [ -e $f ] || continue
          for a in 1 2 3; do
            m1=$(md5sum $f | cut -c1-32); rsync -a "${LD[@]}" $f $sd.partial/ 2>>$RUNLOG; m2=$(md5sum $f | cut -c1-32); ml=$(md5sum $sd.partial/$(basename $f) | cut -c1-32)
            [ "$m1" = "$m2" ] && [ "$ml" = "$m2" ] && { echo "$ml  $(basename $f)" >> $sd.partial/MD5SUMS; break; }
            [ $a = 3 ] && ok=0
          done
        done
        if [ $ok = 1 ]; then echo $SMT > $sd.partial/SRC_MTIME; rm -rf -- $sd; mv $sd.partial $sd; touch $sd/.ok; else md5=bad; note="$note copy/md5 failed;"; fi
      fi
      if [ -e $sd/.ok ] && [ ! -e $sd/CHECK.txt ]; then
        C=$(nice -n 10 $BPY $CHK v6o $sd $P ${OPTS//+/ } 2>&1); echo "$C" > $sd/CHECK.txt
        echo "$C" | awk '$1=="RESUMABLE"{print $2}' > $sd/RESUMABLE
        echo "--resume with --out $RUNDIR --tag $TAG (state ${P}_state.pt; to resume from the mirror copy it back to $RUNDIR)" > $sd/RESUME_ARGS
      fi
      [ -e $sd/.ok ] && md5=ok
      [ "$(cat $sd/RESUMABLE 2>/dev/null)" = yes ] || note="$note $(grep '^WHY' $sd/CHECK.txt 2>/dev/null | cut -c5-);"
      [ -e $(dirname $RUNDIR)/$TAG.done ] && [ $run = no ] && final=yes
    else note="$note no state.pt;"; fi ;;
  qh)
    M=$(grep -E ' start step[0-9]+_m[0-9]+ ' $RUNDIR/chain_status.txt | tail -1 | sed -n 's/.*_m\([0-9]*\) .*/\1/p')
    tail -1 $RUNDIR/chain_status.txt | grep -q 'end step5' && final=yes
    CK=$RUNDIR/qh_member${M}_channels.npz
    if [ -n "$M" ] && [ -e $CK ]; then
      SMT=$(stat -c %Y $CK); sd=$base/m${M}_$SMT
      if [ ! -e $sd/.ok ] && [ $LOWSPACE = 0 ]; then
        prev=$(best_state $base | cut -d' ' -f1); LD=(); [ "$prev" != - ] && LD=(--link-dest=$prev)
        rm -rf -- $sd.partial; mkdir -p $sd.partial; ok=1
        for f in $CK $RUNDIR/qh_member${M}.json $RUNDIR/qh_member${M}_arrays.npz $RUNDIR/chain_status.txt; do
          for a in 1 2 3; do
            m1=$(md5sum $f | cut -c1-32); rsync -a "${LD[@]}" $f $sd.partial/ 2>>$RUNLOG; m2=$(md5sum $f | cut -c1-32); ml=$(md5sum $sd.partial/$(basename $f) | cut -c1-32)
            [ "$m1" = "$m2" ] && [ "$ml" = "$m2" ] && { echo "$ml  $(basename $f)" >> $sd.partial/MD5SUMS; break; }
            [ $a = 3 ] && ok=0
          done
        done
        if [ $ok = 1 ]; then echo $SMT > $sd.partial/SRC_MTIME; rm -rf -- $sd; mv $sd.partial $sd; touch $sd/.ok; else md5=bad; note="$note copy/md5 failed;"; fi
      fi
      if [ -e $sd/.ok ] && [ ! -e $sd/CHECK.txt ]; then
        C=$(nice -n 10 $BPY $CHK qh $sd $M 2>&1); echo "$C" > $sd/CHECK.txt
        echo "$C" | awk '$1=="RESUMABLE"{print $2}' > $sd/RESUMABLE
        echo "rerun $SDIR (members with a json are skipped; the channel loop restarts from qh_member${M}_channels.npz; copy it back to $RUNDIR to resume from the mirror)" > $sd/RESUME_ARGS
      fi
      SCLK=m${M}k$(awk '$1=="LAST"{print $2}' $sd/CHECK.txt 2>/dev/null)
      [ -e $sd/.ok ] && md5=ok
      [ "$(cat $sd/RESUMABLE 2>/dev/null)" = yes ] || note="$note $(grep '^WHY' $sd/CHECK.txt 2>/dev/null | cut -c5-);"
    else note="$note no channel checkpoint yet (member $M);"; fi ;;
  *) note="unknown kind $KIND";;
  esac
  prune $base
  read -r BD BMT BRES <<< "$(best_state $base)"
  if [ "$BD" != - ]; then
    mirr=yes; resm=$BRES; age=$(( (NOW - BMT) / 60 ))
    case $KIND in ladder|twofield) clock=${BD##*/T};; v6o) clock=$(echo ${BD##*/} | sed 's/_[0-9]*$//');; qh) clock=${SCLK:-${BD##*/}};; esac
  fi
  [ "$md5" = "-" ] && [ $mirr = yes ] && md5=ok
  L="$NOWI job=$JOB card=$CARD kind=$KIND running=$run state=$clock age_min=$age mirrored=$mirr md5=$md5 resumable=$resm final=$final"
  [ -n "$note" ] && L="$L note=\"$(echo $note | tr -s ' ')\""
  echo "$L" >> $LOG
  # alert rules
  why=""
  [ "$run" = "?" ] && why="card $CARD unreachable;"
  if [ "$run" = yes ]; then
    [ $mirr = no ] && why="$why running with no mirrored state;"
    [ $mirr = yes ] && [ "$age" -gt $MAXAGE_MIN ] && why="$why newest mirrored resumable state ${age} min old (> ${MAXAGE_MIN});"
  fi
  [ $mirr = yes ] && [ "$resm" != yes ] && why="$why mirrored state not resumable;"
  [ "$run" = no ] && [ $final = no ] && [ $mirr = no ] && why="$why stopped, no mirrored resumable state;"
  [ "$run" = no ] && [ $final = no ] && [ "$EXPECT" = run ] && why="$why process not found and the run is not at its end (stopped or died: relaunch from the mirror);"
  [ -n "$why" ] && ALERTS+=("$JOB ($CARD): $why ${note:+[$note]}")
done < "$CONF"

if [ ${#ALERTS[@]} -gt 0 ]; then
  { echo "busi resume audit ALERT $NOWI (script $HERE/busi_resume_audit_v1.sh, conf $CONF, log $LOG)"; printf '%s\n' "${ALERTS[@]}"; } > $ALERT.tmp && mv $ALERT.tmp $ALERT
else rm -f $ALERT; fi
say "end alerts ${#ALERTS[@]}"
