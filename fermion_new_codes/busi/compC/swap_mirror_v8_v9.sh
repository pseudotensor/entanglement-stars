#!/bin/bash
# swap_mirror_v8_v9.sh (the busi lane, inbox #154/#156): replaces one running c_mirror_v8.sh by c_mirror_v9.sh with the same arguments and log.
# Waits (up to 40 min) until the v8 process group runs a python or a sleep and no scp, ssh, md5sum, mv or rm (never inside a copy, md5 or pod rm), stops that
# group by its own pgid (the mirror was started with setsid), removes a leftover .part in the transit, and starts v9 appended to the same log.
# Usage (box): bash swap_mirror_v8_v9.sh PGID LOG HOST PORT BASE MEMBER RUNTAG RS
set -u
G=${1:?pgid}; L=${2:?log}; shift 2; M=${4:?member}; RT=${5:?runtag}
TR=${QI2_ROOT}/${M}_${RT}
ps -o args= -p $G | grep -q "c_mirror_v8.sh $1 $2 " || { echo "pgid $G is not the v8 mirror of $1 $2"; exit 2; }
for i in $(seq 480); do
  K=$(ps -eo pid=,pgid=,comm= | awk -v g=$G '$2==g && $1!=g {print $3}' | sort -u | tr '\n' ' ')
  if ! echo " $K" | grep -Eq ' (scp|ssh|md5sum|mv|rm) ' && echo " $K" | grep -Eq ' (python|sleep) '; then
    kill -TERM -- -$G; sleep 2; echo "$(date -u '+%F %T UTC') swap: stopped c_mirror_v8 pgid $G (group: $K)" >> "$L"
    for p in "${TR:?}"/*.part; do [ -e "$p" ] && rm -f "${p:?}" && echo "$(date -u '+%F %T UTC') swap: removed leftover $(basename $p)" >> "$L"; done
    cd ${QI2_ROOT}/compC && HF=0 setsid nohup bash c_mirror_v9.sh "$@" >> "$L" 2>&1 < /dev/null &
    sleep 1; echo "swapped pgid $G -> v9 pid $!"; exit 0
  fi
  sleep 5
done
echo "pgid $G never idle in 40 min; not swapped"; exit 1
