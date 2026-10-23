#!/usr/bin/env python
"""build_pod_launch_C_v9.py (busi lane, Computation C at r_s/a 64; busi/specs/C64_spec_v1.md sec. 5): applies the replacement list below to
pod_launch_C_v8.sh and writes pod_launch_C_v9.sh (mode 755); every old string must occur exactly once in the source, otherwise the build aborts
before writing anything. Also writes chain_C64_v1.sh (one card's pair: member, resume-gate test, control).
Usage: build_pod_launch_C_v9.py [pod_launch_C_v8.sh] [pod_launch_C_v9.sh] [chain_C64_v1.sh]"""
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "pod_launch_C_v8.sh"
DST = sys.argv[2] if len(sys.argv) > 2 else "pod_launch_C_v9.sh"
CHAIN = sys.argv[3] if len(sys.argv) > 3 else "chain_C64_v1.sh"

HEADER9 = (
    "#!/bin/bash\n"
    "# pod_launch_C_v9.sh (busi lane): version 8 plus the 4 members of Computation C's averaging control at r_s/a 64 (busi/specs/C64_spec_v1.md;\n"
    "# the owner's funding, inbox #140; REVIEWA_REORIENT_4.md item 11, C5): c64s4 and c64s4off (sigma/a 4, coupling on and off) and c64s8 and\n"
    "# c64s8off (sigma/a 8), each on 2048 sites at --rs 64 (the chain's end at 32 r_s), tau 2 (1/32 r_s per record), prepared at prep-ell 4 and\n"
    "# evolved at ell 4 or 8, the Newton loop and the light arguments of the r_s/a 32 sigma/a 4 member unchanged, to T/r_s 56 (1793 records; the\n"
    "# chain end's return into 16 r_s by the vacuum form is near 56.6, the engine's print on the T = 0 fields the record): complex128 orbitals\n"
    "# slices at every 1 r_s 0 to 56 and light-only slices at the half clocks 10.5 to 55.5; out results/C64, tags C64_*. Version 9 also (i)\n"
    "# computes the gate mode's end clock with the record step tau/rs from RS (1/32 at r_s/a 64; version 8's 2/32 was the r_s/a 32 step), (ii)\n"
    "# refuses a fresh or resumed launch of an r_s/a 64 member below 76600 MiB free on the card (the measured 66.8 GB share, read as 68403 MiB,\n"
    "# plus the 8192 MiB guard; a gate below 42000 MiB), so that no live run is squeezed, and (iii) writes results/C64/LABEL_<tag>.txt at a fresh\n"
    "# launch (the ruling's label, which the engine's head note cannot carry). The r_s/a 32 members are version 8's, unchanged. Version 8's\n"
    "# header follows.\n"
    "# pod_launch_C_v8.sh (busi lane): version 7 plus the continuations s4c80 and s4offc80")

CASE9 = ('COUP=on; TEND=40; RS=32; case $M in '
         'c64s4) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=56;; '
         'c64s4off) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=56;; '
         'c64s8) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=56;; '
         'c64s8off) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=56;; '
         's4) ELL=4;')

SLICES64 = ('  SL4="--slices-c128 $(seq -s, 41 80) --slices-light-c128 $(seq -s, 40.5 1 79.5)"\nfi\n'
            'if [ "$RS" = 64 ]; then   # v9: the r_s/a 64 members\' slice clocks (C64_spec_v1.md sec. 1.5): orbitals every 1 r_s 0 to 56 (restarts), light-only at the half clocks 10.5 to 55.5\n'
            '  SL4="--slices-c128 $(seq -s, 0 56) --slices-light-c128 $(seq -s, 10.5 1 55.5)"\n'
            'fi\n')

GUARD9 = ('stamp=$(date -u +%Y%m%dT%H%M%SZ)\n'
          'if [ "$RS" = 64 ]; then   # v9: an r_s/a 64 member holds the measured r_s/a 32 share (66.8 GB, the same arrays at 2048 sites); nothing fits beside it under the 8192 MiB guard (C64_spec_v1.md sec. 3)\n'
          '  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); NEED=76600; [ "$MODE" = gate ] && NEED=42000\n'
          '  [ "$FREE" -ge "$NEED" ] 2>/dev/null || { echo "refused: $FREE MiB free on the card, $NEED MiB needed for $M in mode $MODE"; echo "C${RS}_$M refused ($FREE MiB free, $NEED needed, mode $MODE) $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; exit 6; }\n'
          'fi\n')

LABEL9 = ('  st "launch $gpuinfo"\n'
          '  if [ "$RS" = 64 ]; then   # v9: the ruling\'s label beside the run files (C64_spec_v1.md sec. 1.6)\n'
          '    cat > $B/results/C64/LABEL_$TAG.txt <<EOF\n'
          'Computation C, the self-consistent 2-field collapse (D16), the averaging control at r_s/a 64 (busi/specs/C64_spec_v1.md): the heavy field the production\'s dust at r_s/a 64 (N_B rs/(2gm) = 640, ma 0.2, g 1/4, the levitating-well preparation at prep-ell 4, R0 4 r_s), the light field the massless 4-component Dirac field of the same operator with multiplicity 16 kappa, kappa 1 and 2 sourcing, kappa 3 the truncation guard; the light reference 1(h_m - mu S < 0) continued by the certified walk (fallback the LZ convention, marked); the chain of 2048 sites, its end at 32 r_s; the chain end\'s first return (the vacuum form from 2 r_s at T = 0; the engine\'s print on the T = 0 fields is the record): into 24 r_s near T/r_s 50, into 16 r_s near 57, into 8 r_s near 63; the formation\'s light reaches 16 r_s about 23 r_s and 24 r_s about 33 r_s after the formation; the run to T/r_s 56: the 16 r_s collar carries chain-independent radiation from about 39 to 56 and the 8 r_s collar from about 28, the 24 r_s collector radius has no chain-independent window on this chain; sigma/a $ELL (evolved) prepared at ell 4; light coupling $COUP; tau 2 (1/32 r_s per record); engine_twofield_v3.py unchanged (its head note\'s "run\'s end at T/rs 40" is the r_s/a 32 chain\'s sentence and does not apply; the numbers beside it do); tag $TAG.\n'
          'EOF\n'
          '  fi\n')

REPL = [
    # 1 the header
    ("#!/bin/bash\n# pod_launch_C_v8.sh (busi lane): version 7 plus the continuations s4c80 and s4offc80", HEADER9),
    # 2 the usage message
    ("M=${1:?member s4 or s2}; MODE=fresh",
     "M=${1:?member s4, s2, s2off, s4off, s4c80, s4offc80, c64s4, c64s4off, c64s8 or c64s8off}; MODE=fresh"),
    # 3 the member table: RS and the 4 r_s/a 64 members in front of version 8's
    ("COUP=on; TEND=40; case $M in s4) ELL=4;", CASE9),
    # 4 the table's refusal message
    ('*) echo "member s4, s2, s2off, s4off, s4c80 or s4offc80"; exit 2;; esac   # v8',
     '*) echo "member s4, s2, s2off, s4off, s4c80, s4offc80, c64s4, c64s4off, c64s8 or c64s8off"; exit 2;; esac   # v9'),
    # 5 the out dir per rs
    ("cd $B || exit 1; mkdir -p results/C32 states logs", "cd $B || exit 1; mkdir -p results/C$RS states logs"),
    # 6 the slice clocks of the r_s/a 64 members
    ('  SL4="--slices-c128 $(seq -s, 41 80) --slices-light-c128 $(seq -s, 40.5 1 79.5)"\nfi\n', SLICES64),
    # 7 COMMON's --rs from RS (everything else identical: C64_spec_v1.md sec. 1.3)
    ('COMMON="--m 0.2 --g 0.25 --rs 32 --R0-over-rs 4', 'COMMON="--m 0.2 --g 0.25 --rs $RS --R0-over-rs 4'),
    # 8 the memory guard
    ("stamp=$(date -u +%Y%m%dT%H%M%SZ)\n", GUARD9),
    # 9 the status prefix
    ('st() { echo "C32_$M $* $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; }',
     'st() { echo "C${RS}_$M $* $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; }'),
    # 10 the label file at a fresh launch
    ('  st "launch $gpuinfo"\n', LABEL9),
    # 11 the fresh launch's out dir and log
    ("--gpu 0 --out $B/results/C32 --tag $TAG > $B/logs/C32_$M.log 2>&1 &",
     "--gpu 0 --out $B/results/C$RS --tag $TAG > $B/logs/C${RS}_$M.log 2>&1 &"),
    # 12 the fresh launch's pid file
    ('  PID=$!; echo $PID > $B/C32_$M.pid; st "pid $PID"\n', '  PID=$!; echo $PID > $B/C${RS}_$M.pid; st "pid $PID"\n'),
    # 13 the snapshot copy
    ("  cp -p $B/results/C32/collapse_${TAG}.json $B/results/C32/collapse_${TAG}_*.npz $SNAP/ || exit 4",
     "  cp -p $B/results/C$RS/collapse_${TAG}.json $B/results/C$RS/collapse_${TAG}_*.npz $SNAP/ || exit 4"),
    # 14 the resume branch's liveness check
    ('  if [ -f $B/C32_$M.pid ] && kill -0 "$(cat $B/C32_$M.pid)" 2>/dev/null; then echo "member $M still running (PID $(cat $B/C32_$M.pid)); not restarted"; exit 5; fi',
     '  if [ -f $B/C${RS}_$M.pid ] && kill -0 "$(cat $B/C${RS}_$M.pid)" 2>/dev/null; then echo "member $M still running (PID $(cat $B/C${RS}_$M.pid)); not restarted"; exit 5; fi'),
    # 15 the resume launch's out dir and log
    ('--gpu 0 --out $B/results/C32 --tag $TAG \\\n    --resume-slice "$SLICE" --resume-run $SNAP/$TAG > $B/logs/C32_${M}_resume_$stamp.log 2>&1 &',
     '--gpu 0 --out $B/results/C$RS --tag $TAG \\\n    --resume-slice "$SLICE" --resume-run $SNAP/$TAG > $B/logs/C${RS}_${M}_resume_$stamp.log 2>&1 &'),
    # 16 the resume launch's pid file
    ('  PID=$!; echo $PID > $B/C32_$M.pid; st "pid $PID (resumed)"', '  PID=$!; echo $PID > $B/C${RS}_$M.pid; st "pid $PID (resumed)"'),
    # 17 the gate's end clock: the record step tau/rs from RS (2/32 was the r_s/a 32 step)
    ('TE=$(python -c "import sys; print(repr(float(sys.argv[1]) + int(sys.argv[2]) * 2.0 / 32.0))" "$TS" "$NREC")',
     'TE=$(python -c "import sys; print(repr(float(sys.argv[1]) + int(sys.argv[2]) * 2.0 / float(sys.argv[3])))" "$TS" "$NREC" "$RS")'),
]

CHAIN_SH = r'''#!/bin/bash
# chain_C64_v1.sh (busi lane, Computation C at r_s/a 64; busi/specs/C64_spec_v1.md sec. 5): one card's pair under pod_launch_C_v9.sh, the member
# then its coupling-off control, with the resume gate run on the finished member (2 records from its slice at T/r_s 55: the restart path tested
# at r_s/a 64 before any continuation needs it). It stops no other process: it waits for the card (76600 MiB free; the launcher's own guard
# refuses otherwise) before every launch and polls every 10 min. An interrupted run (no end rule, last record below T/r_s 56) gets one
# gate-and-resume from its latest slice at or before the last record minus 2 records; a second interruption, a FAIL of the gate, or a member
# ended by the Newton end rule (end_v32) stop the chain with a line in status.txt for the builder's report. A member stopped by P2 (end_s11) is
# complete; its control still runs to 56 (sec. 3 and 6). The builder starts it under nohup after the card's queued work has ended:
#   chain_C64_v1.sh c64s4 c64s4off [BASE]    pod 3, after O5's fourth cube run;   chain_C64_v1.sh c64s8 c64s8off [BASE]    pod 1, after "C32_s4offc80 exit".
MEM=${1:?member}; CTL=${2:?control}; B=${3:-<internal>/busi_compC}; RS=64; TEND=56; L=$B/code/pod_launch_C_v9.sh
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
cs "pair done: member $S_MEM control $S"
'''


def main():
    s = open(SRC).read()
    for i, (old, new) in enumerate(REPL, 1):
        n = s.count(old)
        if n != 1:
            raise SystemExit("build_pod_launch_C_v9: replacement %d: the old string occurs %d times in %s (exactly 1 required); aborted, nothing written:\n%s" % (i, n, SRC, old))
    for i, (old, new) in enumerate(REPL, 1):
        n = s.count(old)
        if n != 1:
            raise SystemExit("build_pod_launch_C_v9: replacement %d: the old string occurs %d times after the earlier replacements; aborted, nothing written:\n%s" % (i, n, old))
        s = s.replace(old, new, 1)
    for must in ("c64s4) RS=64;", "c64s8off) RS=64;", "--rs $RS", "float(sys.argv[3])", 'if [ "$RS" = 64 ]', "NEED=76600", "LABEL_$TAG.txt",
                 "seq -s, 0 56", "seq -s, 10.5 1 55.5", "results/C$RS", "C${RS}_$M"):
        if must not in s:
            raise SystemExit("build_pod_launch_C_v9: the result lacks %r; aborted, nothing written" % must)
    for gone in ("results/C32", "logs/C32", "$B/C32_", '"C32_$M '):
        if gone in s:
            raise SystemExit("build_pod_launch_C_v9: the result still carries %r; aborted, nothing written" % gone)
    chain = CHAIN_SH.replace('cs "pair done: member $S_MEM control $S"', 'cs "pair done: control state $S at T/rs $TL"')
    with open(DST, "w") as fh:
        fh.write(s)
    os.chmod(DST, 0o755)
    with open(CHAIN, "w") as fh:
        fh.write(chain)
    os.chmod(CHAIN, 0o755)
    print("wrote %s (%d replacements on %s, each old exactly once) and %s" % (DST, len(REPL), SRC, CHAIN))


if __name__ == "__main__":
    main()
