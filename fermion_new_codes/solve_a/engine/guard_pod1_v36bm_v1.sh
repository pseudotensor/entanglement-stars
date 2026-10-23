#!/bin/bash
# guard_pod1_v36bm_v1.sh (solve builder 7, analyst; engine spec 48.5 (1)): replaces guard_pod1_A2_v1.sh (stopped by its PID). Every 60 s the
# H100's free memory is read; below 8192 MiB the placed process, the stored-tolerance member (engine_collapse_v36b.py ... the tag in
# ${QI2_ROOT}/CURRENT_TAG), is stopped by its PID (command line read first; its 2 pair workers, PIDs from its log, stopped
# after it if still alive) and re-queued: relaunched from its latest complex128 slice through the slice restart (launch_v36bm_pod1_v1.sh SLICE
# RUN N: --resume-slice, --resume-run) once 45000 MiB are free. A2 (page_history_v6i ... v6iA2_B2) is never stopped. Every 10 min every
# process's device share is printed. Ends when the member ends by itself (or is stopped by its switch: no relaunch then).
# Usage (pod 1): setsid nohup bash guard_pod1_v36bm_v1.sh > /dev/null 2>&1 < /dev/null &
D=${QI2_ROOT}/v36b_member; L=$D/guard_v1.log; C=${QI2_ROOT}/code
log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" >> $L; }
mpid() { local t=$(cat $D/CURRENT_TAG 2>/dev/null); [ -n "$t" ] && ps -eo pid,args | grep "[e]ngine_collapse_v36b.py.*--tag $t\b" | grep -v "bash " | awk '{print $1}' | head -1; }
shares() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';'; }
log "guard v1 start; member PID $(mpid) tag $(cat $D/CURRENT_TAG 2>/dev/null)"
STOPPED=0; n=0; NR=$(ls $D/collapse_*_v36b_r*.json 2>/dev/null | wc -l)
while true; do
  P=$(mpid); FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ -z "$P" ]; then
    if [ $STOPPED -eq 1 ]; then
      if [ "${FREE:-0}" -ge 45000 ]; then
        T=$(cat $D/CURRENT_TAG); S=$(ls -1 $D/states/$T/orbitals_${T}_T*.npz 2>/dev/null | sort | tail -1)
        if [ -z "$S" ]; then
          NR=$((NR + 1)); log "no slice of $T on the pod: relaunched from T = 0 is not done by the guard; left for the builder"; break
        fi
        NR=$((NR + 1))
        (cd $D && setsid nohup bash $C/launch_v36bm_pod1_v1.sh $S $D/$T $NR > /dev/null 2>&1 < /dev/null &); sleep 60; STOPPED=0
        log "member re-queued: relaunched from $(basename $S) of $T as tag $(cat $D/CURRENT_TAG) at ${FREE} MiB free; PID $(mpid)"
      fi
    else
      log "the member ended by itself or by its switch: $(tail -1 $D/status.txt); guard ends"; break
    fi
  elif [ "${FREE:-99999}" -lt 8192 ]; then
    T=$(cat $D/CURRENT_TAG)
    if tr '\0' ' ' < /proc/$P/cmdline | grep -q "engine_collapse_v36b.py.*--tag $T"; then
      W=$(grep -o "pair workers started (pids [0-9,]*" $D/$T.log | tail -1 | grep -o "[0-9,]*$" | tr ',' ' ')
      kill $P; STOPPED=1; log "GUARD: free ${FREE} MiB below 8192: the member stopped by PID $P; shares: $(shares)"
      sleep 20
      for w in $W; do [ -e /proc/$w ] && tr '\0' ' ' < /proc/$w/cmdline | grep -q multiprocessing && kill $w && log "pair worker $w stopped"; done
    fi
  fi
  [ $((n % 10)) -eq 0 ] && log "free ${FREE} MiB; member PID ${P:-none}; shares: $(shares)"
  n=$((n + 1)); sleep 60
done
