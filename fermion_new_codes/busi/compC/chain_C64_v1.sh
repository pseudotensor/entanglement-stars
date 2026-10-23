#!/bin/bash
# chain_C64_v1.sh (busi lane, Computation C at r_s/a 64; busi/specs/C64_spec_v1.md sec. 5): one card's pair under pod_launch_C_v9.sh, the member
# then its coupling-off control, with the resume gate run on the finished member (2 records from its slice at T/r_s 55: the restart path tested
# at r_s/a 64 before any continuation needs it). It stops no other process: it waits for the card (76600 MiB free; the launcher's own guard
# refuses otherwise) before every launch and polls every 10 min. An interrupted run (no end rule, last record below T/r_s 56) gets one
# gate-and-resume from its latest slice at or before the last record minus 2 records; a second interruption, a FAIL of the gate, or a member
# ended by the Newton end rule (end_v32) stop the chain with a line in status.txt for the builder's report. A member stopped by P2 (end_s11) is
# complete; its control still runs to 56 (sec. 3 and 6). The builder starts it under nohup after the card's queued work has ended:
#   chain_C64_v1.sh c64s4 c64s4off [BASE]    pod 3, after O5's fourth cube run;   chain_C64_v1.sh c64s8 c64s8off [BASE]    pod 1, after "C32_s4offc80 exit".
MEM=${1:?member}; CTL=${2:?control}; B=${3:-${QI2_ROOT}/busi_compC}; RS=64; TEND=56; L=$B/code/pod_launch_C_v9.sh
cd $B || exit 1; mkdir -p results/C64 states logs gate snap
tag_of() { case $1 in c64s4) echo C64_s4_ell4_tau2_twofield_v3;; c64s4off) echo C64_s4_ell4_tau2_twofield_v3_off;; c64s8) echo C64_s8_ell8_tau2_twofield_v3;; c64s8off) echo C64_s8_ell8_tau2_twofield_v3_off;; *) echo "";; esac; }
[ -n "$(tag_of $MEM)" ] && [ -n "$(tag_of $CTL)" ] || { echo "members: c64s4 c64s4off or c64s8 c64s8off"; exit 2; }
cs() { echo "chain_C64 $MEM/$CTL $* $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; echo "$*"; }
wait_card() { while :; do F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$F" -ge ${1:-76600} ] 2>/dev/null && return 0; sleep 600; done; }
state_of() {  # prints: done|ended_v32|ended_s11|interrupted <last T/rs>
  python - "$B/results/C64/collapse_$1.json" "$TEND" <<'EOF'
import json, sys
try:
    J = json.load(open(sys.argv[1]))
except Exception:
    print("interrupted -1"); sys.exit(0)
recs = [r for r in J.get("records", []) if r.get("E") is not None]
TL = recs[-1]["T_over_rs"] if recs else -1.0
if "end_v32" in J: print("ended_v32 %r" % TL)
elif "end_s11" in J: print("ended_s11 %r" % TL)
elif TL >= float(sys.argv[2]) - 1e-9: print("done %r" % TL)
else: print("interrupted %r" % TL)
EOF
}
pick_slice() {  # the latest orbitals slice of tag $1 on the pod at a clock <= $2 (T/rs)
  python - "$B/states" "$1" "$2" <<'EOF'
import glob, os, sys
fs = sorted(glob.glob(os.path.join(sys.argv[1], "orbitals_%s_T*.npz" % sys.argv[2])))
ok = [f for f in fs if float(f.rsplit("_T", 1)[1][:-4]) <= float(sys.argv[3]) + 1e-9]
print(ok[-1] if ok else "")
EOF
}
gate_verdict() { tail -1 "$(ls -td $B/gate/*_$1 | head -1)/resume_gate.log" 2>/dev/null; }
run_one() {  # $1 the member name; sets S and TL; returns nonzero where the chain must stop
  local T SL V; T=$(tag_of $1)
  wait_card 76600; cs "$1 launch"; bash $L $1 $B; cs "$1 launcher exit $?"
  read S TL <<<"$(state_of $T)"; cs "$1 state $S at T/rs $TL"
  if [ "$S" = interrupted ]; then
    SL=$(pick_slice $T $(python -c "print(float('$TL') - 4.0 / $RS)")); [ -n "$SL" ] || { cs "$1 interrupted with no slice on the pod at or before T/rs $TL minus 2 records; STOP (copy a slice back from the box, then gate and resume by hand)"; return 7; }
    wait_card 42000; bash $L $1 gate "$SL" 2 $B; V=$(gate_verdict $1); cs "$1 resume gate from $SL: $V"
    echo "$V" | grep -Eq 'BITWISE|WITHIN-FLOOR' || { cs "$1 resume gate FAIL; STOP"; return 8; }
    wait_card 76600; cs "$1 resume from $SL"; bash $L $1 resume "$SL" $B; cs "$1 resumed launcher exit $?"
    read S TL <<<"$(state_of $T)"; cs "$1 state after the resume $S at T/rs $TL"
    [ "$S" = interrupted ] && { cs "$1 interrupted twice; STOP"; return 9; }
  fi
  return 0
}
run_one $MEM || exit $?
[ "$S" = ended_v32 ] && { cs "the member ended by the Newton end rule (end_v32) at T/rs $TL; the control is not launched; report"; exit 10; }
[ "$S" = ended_s11 ] && cs "the member stopped by P2 (end_s11) at T/rs $TL; the control runs to $TEND (C64_spec_v1.md sec. 3)"
# the restart path tested at r_s/a 64: 2 records from the finished member's slice at T/rs 55 (or the latest at or before it on the pod)
TM=$(tag_of $MEM); SL=$(pick_slice $TM $((TEND - 1)))
if [ -n "$SL" ]; then wait_card 42000; bash $L $MEM gate "$SL" 2 $B; V=$(gate_verdict $MEM); cs "member resume gate (the r_s/a 64 test) from $SL: $V"
else cs "member resume gate (the r_s/a 64 test): no slice at or before T/rs $((TEND - 1)) on the pod; the test runs by hand from a slice copied back"; fi
run_one $CTL || exit $?
cs "pair done: control state $S at T/rs $TL"
