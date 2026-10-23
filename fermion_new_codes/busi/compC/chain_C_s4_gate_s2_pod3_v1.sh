#!/bin/bash
# pod 3 (the busi lane's H100; the owner: the pod always has critical work queued): when C's sigma/a 4 member ends, (1) the resume gate of the
# resumability audit (inbox #86): pod_launch_C_v3.sh s4 gate on its T/r_s 36 orbitals slice for 2 records, against the stored records; (2) C's
# sigma/a 2 member from T = 0 on this card (pod_launch_C_v3.sh s2, the arguments of version 2) with its guard. The T/r_s 36 slice is hard-linked
# into BASE/gate_slices/ the moment the member ends, because c_mirror_v3 moves every slice of an ended member to the box and deletes the pod copy;
# the link is removed after the gate. If the sigma/a 4 member does not end with exit 0 (a guard stop or a failure), nothing is launched and the
# chain says so in its log: the restart is then decided by hand. Never touches any process but its own children.
B=${1:-${QI2_ROOT}/busi_compC}; L=$B/logs/chain_s4_gate_s2.log; TAG=C32_s4_ell4_tau2_twofield_v3
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
PID=$(cat $B/C32_s4.pid 2>/dev/null)
log "chain start (pid $$): waiting for C32_s4 pid $PID"
while [ -n "$PID" ] && kill -0 $PID 2>/dev/null; do sleep 30; done
mkdir -p $B/gate_slices
S36=$B/states/orbitals_${TAG}_T0036.00.npz; G36=$B/gate_slices/orbitals_${TAG}_T0036.00.npz
[ -f $S36 ] && ln -f $S36 $G36 && log "slice T/rs 36 linked for the gate"
for i in $(seq 1 20); do grep -q "C32_s4 exit" $B/status.txt && break; sleep 3; done
EX=$(grep "C32_s4 exit\|C32_s4 STOPPED" $B/status.txt | tail -1)
log "C32_s4 ended: $EX"
case "$EX" in *"exit 0 "*) ;; *) log "the sigma/a 4 member did not end with exit 0: no gate, no sigma/a 2 launch; decide by hand"; exit 1;; esac
if [ -f $G36 ]; then
  log "resume gate starts (2 records from T/rs 36)"
  bash $B/code/pod_launch_C_v3.sh s4 gate $G36 2 $B >> $L 2>&1
  log "resume gate exit $?: $(grep 'resume gate compare' $B/status.txt | tail -1)"
  rm -f $G36
else
  log "no T/rs 36 slice on the pod: the gate is skipped"
fi
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 15; done
log "C32_s2 launch"
(setsid nohup bash $B/code/pod_launch_C_v3.sh s2 $B > $B/launch_s2.out 2>&1 < /dev/null &)
sleep 30
(setsid nohup bash $B/code/pod_guard_C_v2.sh s2 $B > /dev/null 2>&1 < /dev/null &)
log "C32_s2 pid $(cat $B/C32_s2.pid 2>/dev/null) with its guard; chain ends"
