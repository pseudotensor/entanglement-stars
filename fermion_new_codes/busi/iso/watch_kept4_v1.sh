#!/bin/bash
# watch_kept4_v1.sh MEMBER [N] [THR] [W] [ONE_THR] (busi 2026-10-19, early_failures_ruling_v3 4.2; also the retry's watch of 2.2).
# Watches one ladder member's engine (pid from $R/pids/MEMBER.pid, its cmdline verified: engine_collapse_v36*.py and --tag TAG naming MEMBER)
# and its log $R/logs/TAG.log. Each record line '^T/rs X ...' has a kept residual: the value in 'UNCONVERGED (residual R)' when printed, else
# 0 (converged). The rule, tested at every record in log order: N of the last W records with a kept residual above THR (defaults N 4, THR 1e-3,
# W = N, i.e. four consecutive), or, when ONE_THR is given, one record above ONE_THR. On the rule: the member's one-member ladder queue loop
# ('bash .../launch_ladder_v36d_v1*.sh one_MEMBER queue', exact argument match on ps output) is ended first so its one resume cannot relaunch
# the member, then the engine by verified PID (TERM, KILL after 30 s); the T that fired the rule, the window and the last record at or below
# THR before the window go to $R/watch_kept4_MEMBER.log, the member's status file and $R/STATUS. Exits when the engine exits. No pkill/pgrep -f.
# Run detached: setsid nohup bash watch_kept4_v1.sh MEMBER [N] [THR] [W] [ONE_THR] > /dev/null 2>&1 < /dev/null &
# Dry run on a log (no process touched): bash watch_kept4_v1.sh --dry LOG [N] [THR] [W] [ONE_THR]
M=${1:?member}; N=${2:-4}; THR=${3:-1e-3}; W=${4:-$N}; ONE=${5:-}
R=${LADDER_ROOT:-${QI2_ROOT}/ladders}
rule_of() {   # $1 log: prints 'FIRE T | ...' at the first record (in log order) where the rule holds, else nothing
  grep '^T/rs ' "$1" 2>/dev/null | awk -v n=$N -v thr=$THR -v w=$W -v one="$ONE" '
    { t=$2; r=0; if (match($0, /UNCONVERGED \(residual [^)]*\)/)) { s=substr($0, RSTART+22, RLENGTH-23); r=s+0 }
      k++; T[k]=t; Rv[k]=r
      c=0; lst=""; for (i = (k-w+1 > 1 ? k-w+1 : 1); i <= k; i++) { if (Rv[i] > thr+0) c++; lst = lst " " T[i] ":" (Rv[i] > 0 ? sprintf("%.1e", Rv[i]) : "conv") }
      if (c >= n) { print "FIRE " t " | window" lst " | last record at or below threshold before the window: " (lastok == "" ? "none" : lastok); exit }
      if (one != "" && r > one+0) { print "FIRE " t " | one record above " one " (" sprintf("%.1e", r) ") | window" lst; exit }
      if (k-w+1 >= 1 && Rv[k-w+1] <= thr+0) lastok=T[k-w+1] }'
}
if [ "$M" = --dry ]; then LG=${2:?log}; N=${3:-4}; THR=${4:-1e-3}; W=${5:-$N}; ONE=${6:-}; V=$(rule_of "$LG"); echo "${V:-no fire} (records $(grep -c '^T/rs ' "$LG"); rule $N of $W above $THR${ONE:+, or one above $ONE})"; exit 0; fi
PF=$R/pids/$M.pid; S=$R/status_$M.txt; WL=$R/watch_kept4_$M.log
now() { date -u +%FT%TZ; }
say() { echo "$(now) $M watch_kept4: $*" | tee -a $WL >> $S; }
P=$(cat $PF 2>/dev/null); [ -n "$P" ] && kill -0 "$P" 2>/dev/null || { echo "$(now) $M watch_kept4: no live pid in $PF" >> $WL; exit 1; }
CMD=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null)
TAG=$(tr '\0' '\n' < /proc/$P/cmdline | grep -A1 -x -- --tag | tail -1)
case "$CMD" in *engine_collapse_v36*.py*) ;; *) echo "$(now) $M watch_kept4: pid $P is not an engine ($CMD)" >> $WL; exit 1;; esac
case "$TAG" in *_${M}_*) ;; *) echo "$(now) $M watch_kept4: pid $P tag '$TAG' does not name $M" >> $WL; exit 1;; esac
L=$R/logs/$TAG.log
say "attached: pid $P tag $TAG; rule $N of the last $W kept residuals above $THR${ONE:+, or one above $ONE}; log $L"
verify() { local c; c=$(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null); case "$c" in *engine_collapse_v36*.py*--tag\ $TAG*) return 0;; *) return 1;; esac; }
while kill -0 $P 2>/dev/null; do
  V=$(rule_of $L)
  if [ "${V%% *}" = FIRE ]; then
    say "RULE at T/rs ${V#FIRE }"
    for q in $(ps -eo pid=,args= | awk -v c="one_$M" '$2=="bash" && $3 ~ /launch_ladder_v36d_v1[a-z]*\.sh$/ && $4==c && $5=="queue" {print $1}'); do
      kill -TERM "$q" && say "the one-member queue loop pid $q ended first (no automatic resume)"
    done
    if verify; then kill -TERM $P; say "engine pid $P sent TERM"; sleep 30; kill -0 $P 2>/dev/null && verify && { kill -KILL $P; say "engine pid $P sent KILL"; }
    else say "pid $P no longer the engine; nothing killed"; fi
    echo "$(now) $M ended by watch_kept4 rule at T/rs ${V#FIRE }" >> $R/STATUS; break
  fi
  sleep 60
done
say "exit (engine pid $P gone)"
