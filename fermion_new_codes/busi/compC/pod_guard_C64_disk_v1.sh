#!/bin/bash
# pod_guard_C64_disk_v1.sh (busi lane, pod side; inbox busi #150): a disk guard sized for pod 3's 40 GB container disk. Every 30 s it reads
# the live r_s/a 64 member's PID (the newest BASE/C64_*.pid whose process is alive) and the free space on /: below LOW GB (default 8, two
# 3.5 GB slices) it pauses that process (SIGSTOP), and resumes it (SIGCONT) at HIGH GB (default 12), so that the mirror drains slices first
# and no write fails. It stops nothing else and exits when no C64 member has been alive for 30 min. Log BASE/logs/guard_C64_disk.log.
B=${1:-${QI2_ROOT}/busi_compC}; LOW=${2:-8}; HIGH=${3:-12}; L=$B/logs/guard_C64_disk.log; idle=0; paused=""
log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
log "start low $LOW GB high $HIGH GB"
while :; do
  P=""; for f in $(ls -t $B/C64_*.pid 2>/dev/null); do q=$(cat $f); kill -0 "$q" 2>/dev/null && { P=$q; break; }; done
  if [ -z "$P" ]; then idle=$((idle+1)); [ $idle -ge 60 ] && { log "no live C64 member for 30 min: guard ends"; exit 0; }; sleep 30; continue; fi
  idle=0; F=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$paused" != "$P" ] && [ "${F:-99}" -lt $LOW ]; then kill -STOP $P && paused=$P && log "paused $P (free $F GB)"; fi
  if [ "$paused" = "$P" ] && [ "${F:-0}" -ge $HIGH ]; then kill -CONT $P && paused="" && log "resumed $P (free $F GB)"; fi
  sleep 30
done
