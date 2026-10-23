#!/bin/bash
# yield_at_next_slice_v1.sh (busi, GATECOMP, #408 item: the owner's merge_for_vllm.py may need h100-f): ON THE CARD, run only on the AWS
# agent's or the lead's word. Waits for the member's next slice (newer than the newest now, size stable 60 s), then stops the engine python
# and its pair workers by PID (TERM, KILL after 120 s), and logs where it stopped; the member resumes later from that slice (version 34).
# usage: yield_at_next_slice_v1.sh MEMBER_DIR    (e.g. ${QI2_ROOT}/W256_s4)
set -u
M=$1; ST=$M/states; log() { echo "$(date -u +%FT%TZ) $*" | tee -a $M/YIELD.txt; }
for f in pid_T3 pid; do [ -e $M/run/$f ] && { SH=$(cat $M/run/$f); break; }; done
EP=$(ps -o pid=,cmd= --ppid $SH | awk '/engine_collapse/{print $1; exit}'); [ -n "$EP" ] || { log "no engine pid under $SH"; exit 9; }
N0=$(ls -1t $ST/orbitals_*.npz 2>/dev/null | head -1); log "yield requested; engine pid $EP; newest slice now $(basename ${N0:-none})"
until N=$(ls -1t $ST/orbitals_*.npz 2>/dev/null | head -1) && [ -n "$N" ] && [ "$N" != "$N0" ] && s1=$(stat -L -c %s $N) && sleep 60 && [ "$s1" = "$(stat -L -c %s $N)" ]; do sleep 30; done
log "new slice $(basename $N) complete, md5 $(md5sum < $N | cut -c1-32); stopping engine $EP"
KIDS=$(ps -o pid= --ppid $EP | tr -s ' \n' ' '); kill -TERM $EP
for i in $(seq 1 24); do kill -0 $EP 2>/dev/null || break; sleep 5; done; kill -0 $EP 2>/dev/null && { kill -KILL $EP; log "KILL sent"; }
for k in $KIDS; do kill -0 $k 2>/dev/null && kill -TERM $k; done
log "yielded at $(basename $N); last record $(grep -E '^T/rs [0-9.]+ E/E0' $M/run/log.txt | tail -1 | cut -c1-20); resume: --resume-slice $N --resume-run <the run's json dir/tag>"
