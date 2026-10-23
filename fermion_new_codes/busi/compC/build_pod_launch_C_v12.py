#!/usr/bin/env python
"""build_pod_launch_C_v12.py (the busi lane, Computation C; the transport ruling busi/notes/C64_transport_ruling_v1.md 4.5): writes
pod_launch_C_v12.sh from pod_launch_C_v11.sh (input 09) by exact string replacements, each old string occurring exactly once in version 11 or the
build stops and writes nothing; bash -n checks the result when bash is at hand. What version 12 adds (its header says it in full): the engine
engine_twofield_v9.py for every member with --cut-transport $CUT (record on the version 3 members, sector on the L members; the word record or
sector after a mode's arguments overrides, in either order with implicit|lagged; the R3 guard extended to the word sector on a version 3 member
outside the gate and segment modes); TEND_OVERRIDE for the clock of a launch (5.1(b)); the resume, branch, segment and gate modes at a transport
change (the snapshot copy's args.cut_transport rewritten to this launch's, the old value kept as args.cut_transport_before, the LABEL appended and
status.txt saying so with the change clock); the members c32s4Lsec (5.6) and c64s8Lrec (5.9); the segment and gate directories carrying the
transport word; the LABEL carrying 4.3's sentence on a sector launch and version 9 in the engine's name.
Usage: python build_pod_launch_C_v12.py --v11 PATH/pod_launch_C_v11.sh --out PATH/pod_launch_C_v12.sh"""
import argparse
import shutil
import subprocess


def _s(x):
    """the snippets below start on the line after the opening quotes: that first newline is not part of them."""
    return x[1:] if x.startswith("\n") else x


R = []   # (name, old, new): every old occurs exactly once in version 11

# 1. the header
R.append(("header", _s(r'''
#!/bin/bash
# pod_launch_C_v11.sh (busi lane): the fix ruling's pod_launch_C_v10 (busi/notes/C64_fix_ruling_v4.md, inbox busi #193 item 2), filed as version'''), _s(r'''
#!/bin/bash
# pod_launch_C_v12.sh (busi lane): the transport ruling's launcher (busi/notes/C64_transport_ruling_v1.md 4.5), built over version 11 by
# build_pod_launch_C_v12.py (each old string once, else it aborts). Version 12: (i) the engine is engine_twofield_v9.py (the sector walk;
# busi/code/compC) for every member, with --cut-transport $CUT on every launch: record on the version 3 members (version 3 bitwise, G0/G1 and G0'),
# sector on the lagged (L) members; the word record or sector after a mode's arguments (in either order with implicit|lagged) overrides, so the
# record-scale segments G3 and G3b stay reproducible (c64s4 segment ... lagged, or ... lagged record) and the gates G4, G4c, G4s8 run the version
# 3 members' slices on the sector walk (c64s4 segment ... lagged sector); the red team's R3 guard extended: a version 3 member never takes sector
# outside the gate and segment modes; (ii) TEND_OVERRIDE=X sets --T-over-rs X (the ruling's 5.1(b): the sigma/a 8 member's approach to the clock
# 13.0), recorded in the LABEL and in the json's args.T_over_rs; (iii) the resume, branch, segment and gate modes at a transport change (the run's
# --cut-transport, record when its json carries none, differing from this launch's): the snapshot copy's args.cut_transport is rewritten to this
# launch's as R4 rewrites tau, the old value kept as args.cut_transport_before, the archive untouched, the LABEL (appended on a resume) and
# status.txt saying so with the change clock (the engine's resume check, light_resume_v36, compares the argument); (iv) 2 members added:
# c32s4Lsec (the r_s/a 32 transport control of 5.6: c32s4L's run branched on the sector walk from the slice before its first light fallback,
# tag C32_s4_ell4_tau2_twofield_v4L_sec, out results/C32v4L, TEND 56) and c64s8Lrec (the record-scale branch of 5.9: c64s8L's run branched from
# its 13.0 slice on --cut-transport record to 24, tag C64_s8_ell8_tau2_twofield_v4L_rec, out results/C64v4L); (v) the segment and gate
# directories carry the transport word (gate/segment_<member>_<rsolve>_<cut>_<stamp>/, gate/<stamp>_<member>_<rsolve>_<cut>/); (vi) the LABEL
# carries the transport ruling's 4.3 sentence on a sector launch and names engine_twofield_v9. The modes:
#   pod_launch_C_v12.sh MEMBER [record|sector] [BASE]                                 a fresh launch (TEND_OVERRIDE=X for the clock X)
#   pod_launch_C_v12.sh MEMBER resume SLICE [record|sector] [BASE]                    the restart into the member's own run (a transport change rewrites the snapshot copy)
#   pod_launch_C_v12.sh MEMBER branch SLICE [record|sector] [BASE]                    a new run from a source run's slice (c32s4L, c32s4offL, c32s4Lt1 from version 3's; c32s4Lsec, c64s8Lrec at a transport change)
#   pod_launch_C_v12.sh MEMBER segment SLICE END [lagged|implicit] [record|sector] [BASE]
#   pod_launch_C_v12.sh MEMBER gate SLICE [N] [lagged|implicit] [record|sector] [BASE]
# Everything else is version 11's, unchanged (the memory guards, the slice clocks, the argument sets). Version 11's header follows.
# pod_launch_C_v11.sh (busi lane): the fix ruling's pod_launch_C_v10 (busi/notes/C64_fix_ruling_v4.md, inbox busi #193 item 2), filed as version''')))

# 2. the members' list and the words' parsing
R.append(("members list", _s(r'''
M=${1:?member s4, s2, s2off, s4off, s4c80, s4offc80, c64s4, c64s4off, c64s8, c64s8off, c64s4L, c64s4offL, c64s8L, c64s8offL, c64s4Lt1, c32s4L, c32s4offL or c32s4Lt1}; MODE=fresh   # v11'''), _s(r'''
M=${1:?member s4, s2, s2off, s4off, s4c80, s4offc80, c64s4, c64s4off, c64s8, c64s8off, c64s4L, c64s4offL, c64s8L, c64s8offL, c64s4Lt1, c32s4L, c32s4offL, c32s4Lt1, c32s4Lsec or c64s8Lrec}; MODE=fresh   # v11   # v12: c32s4Lsec, c64s8Lrec''')))
R.append(("words", _s(r'''
RSOLVE=; case ${1:-} in implicit|lagged) RSOLVE=$1; shift;; esac   # v11: the reference solve of a version 3 member's gate or segment (implicit when absent); a lagged member always runs lagged'''), _s(r'''
RSOLVE=; CUTARG=; for _w in 1 2; do case ${1:-} in implicit|lagged) RSOLVE=$1; shift;; record|sector) CUTARG=$1; shift;; esac; done   # v11: the reference solve of a version 3 member's gate or segment (implicit when absent); a lagged member always runs lagged   # v12: the cut's transport word (record|sector) after the mode's arguments, in either order with the reference solve's''')))

# 3. the members' case, the transport and the clock override
R.append(("members case", _s(r'''
COUP=on; TEND=40; RS=32; TAU=2; LAG=0; SRCOUT=; SRCTAG=; case $M in c64s4) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=56;; c64s4off) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=56;; c64s8) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=56;; c64s8off) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=56;; s4) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3; JVP="--nk-jvp exact";; s2) ELL=2; TAG=C32_s2_ell2_tau2_twofield_v3_fd; JVP="";; s2off) ELL=2; TAG=C32_s2_ell2_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off;; s4off) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off;; s4c80) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=80;; s4offc80) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=80;; c64s4L) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v4L; JVP="--nk-jvp exact"; TEND=56; LAG=1;; c64s4offL) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v4L_off; JVP="--nk-jvp exact"; COUP=off; TEND=56; LAG=1;; c64s8L) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v4L; JVP="--nk-jvp exact"; TEND=56; LAG=1;; c64s8offL) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v4L_off; JVP="--nk-jvp exact"; COUP=off; TEND=56; LAG=1;; c64s4Lt1) RS=64; ELL=4; TAG=C64_s4_ell4_tau1_twofield_v4L; JVP="--nk-jvp exact"; TEND=16; TAU=1; LAG=1;; c32s4L) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v4L; JVP="--nk-jvp exact"; TEND=56; LAG=1; SRCOUT=results/C32; SRCTAG=C32_s4_ell4_tau2_twofield_v3;; c32s4offL) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v4L_off; JVP="--nk-jvp exact"; COUP=off; TEND=56; LAG=1; SRCOUT=results/C32; SRCTAG=C32_s4_ell4_tau2_twofield_v3_off;; c32s4Lt1) ELL=4; TAG=C32_s4_ell4_tau1_twofield_v4L; JVP="--nk-jvp exact"; TEND=24; TAU=1; LAG=1; SRCOUT=results/C32; SRCTAG=C32_s4_ell4_tau2_twofield_v3;; *) echo "member s4, s2, s2off, s4off, s4c80, s4offc80, c64s4, c64s4off, c64s8, c64s8off, c64s4L, c64s4offL, c64s8L, c64s8offL, c64s4Lt1, c32s4L, c32s4offL or c32s4Lt1"; exit 2;; esac   # v9   # v11: TAU and LAG per member, the version 3 run a branch starts from, the 8 lagged members'''), _s(r'''
COUP=on; TEND=40; RS=32; TAU=2; LAG=0; SRCOUT=; SRCTAG=; CUTDEF=; case $M in c64s4) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=56;; c64s4off) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=56;; c64s8) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=56;; c64s8off) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=56;; s4) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3; JVP="--nk-jvp exact";; s2) ELL=2; TAG=C32_s2_ell2_tau2_twofield_v3_fd; JVP="";; s2off) ELL=2; TAG=C32_s2_ell2_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off;; s4off) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off;; s4c80) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=80;; s4offc80) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=80;; c64s4L) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v4L; JVP="--nk-jvp exact"; TEND=56; LAG=1;; c64s4offL) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v4L_off; JVP="--nk-jvp exact"; COUP=off; TEND=56; LAG=1;; c64s8L) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v4L; JVP="--nk-jvp exact"; TEND=56; LAG=1;; c64s8offL) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v4L_off; JVP="--nk-jvp exact"; COUP=off; TEND=56; LAG=1;; c64s4Lt1) RS=64; ELL=4; TAG=C64_s4_ell4_tau1_twofield_v4L; JVP="--nk-jvp exact"; TEND=16; TAU=1; LAG=1;; c32s4L) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v4L; JVP="--nk-jvp exact"; TEND=56; LAG=1; SRCOUT=results/C32; SRCTAG=C32_s4_ell4_tau2_twofield_v3;; c32s4offL) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v4L_off; JVP="--nk-jvp exact"; COUP=off; TEND=56; LAG=1; SRCOUT=results/C32; SRCTAG=C32_s4_ell4_tau2_twofield_v3_off;; c32s4Lt1) ELL=4; TAG=C32_s4_ell4_tau1_twofield_v4L; JVP="--nk-jvp exact"; TEND=24; TAU=1; LAG=1; SRCOUT=results/C32; SRCTAG=C32_s4_ell4_tau2_twofield_v3;; c32s4Lsec) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v4L_sec; JVP="--nk-jvp exact"; TEND=56; LAG=1; CUTDEF=sector; SRCOUT=results/C32v4L; SRCTAG=C32_s4_ell4_tau2_twofield_v4L;; c64s8Lrec) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v4L_rec; JVP="--nk-jvp exact"; TEND=24; LAG=1; CUTDEF=record; SRCOUT=results/C64v4L; SRCTAG=C64_s8_ell8_tau2_twofield_v4L;; *) echo "member s4, s2, s2off, s4off, s4c80, s4offc80, c64s4, c64s4off, c64s8, c64s8off, c64s4L, c64s4offL, c64s8L, c64s8offL, c64s4Lt1, c32s4L, c32s4offL, c32s4Lt1, c32s4Lsec or c64s8Lrec"; exit 2;; esac   # v9   # v11: TAU and LAG per member, the version 3 run a branch starts from, the 8 lagged members   # v12: c32s4Lsec (the r_s/a 32 transport control, the ruling's 5.6, branched from c32s4L's run on the sector walk) and c64s8Lrec (the record-scale branch of c64s8L to 24, the ruling's 5.9)
CUT=record; [ "$LAG" = 1 ] && CUT=sector; [ -n "$CUTDEF" ] && CUT=$CUTDEF; [ -n "$CUTARG" ] && CUT=$CUTARG   # v12: the cut's transport: record on the version 3 members, sector on the L members (c64s8Lrec record by its definition), the override word last
TEND=${TEND_OVERRIDE:-$TEND}   # v12: TEND_OVERRIDE (the transport ruling 5.1(b)): --T-over-rs of this launch, recorded in the LABEL and the json's args.T_over_rs (the sigma/a 8 member's approach to 13.0)''')))

# 4. the R3 guard extended to the transport word
R.append(("R3 guard", _s(r'''
if [ "$LAG" = 0 ] && [ "$RSOLVE" = lagged ] && [ "$MODE" != gate ] && [ "$MODE" != segment ]; then echo "$M is a version 3 member: the word lagged applies to its gate and segment modes only (a fresh or resumed lagged run under a version 3 tag and LABEL would write the lagged scheme into the implicit history's files; the ruling: no version 3 run continues); use a lagged member"; exit 2; fi   # v11 (red team R3)'''), _s(r'''
if [ "$LAG" = 0 ] && [ "$RSOLVE" = lagged ] && [ "$MODE" != gate ] && [ "$MODE" != segment ]; then echo "$M is a version 3 member: the word lagged applies to its gate and segment modes only (a fresh or resumed lagged run under a version 3 tag and LABEL would write the lagged scheme into the implicit history's files; the ruling: no version 3 run continues); use a lagged member"; exit 2; fi   # v11 (red team R3)
if [ "$LAG" = 0 ] && [ "$CUT" = sector ] && [ "$MODE" != gate ] && [ "$MODE" != segment ]; then echo "$M is a version 3 member: the word sector applies to its gate and segment modes only (a sector-walk run under a version 3 tag and LABEL would write the sector transport into the record-scale history's files); use a lagged member"; exit 2; fi   # v12 (red team R3 extended to the transport word; the transport ruling 4.5)''')))

# 5. the branch's source message
R.append(("branch source", _s(r'''
BRANCHOUT=$SRCOUT; BRANCHTAG=$SRCTAG; SRCOUT=$OUT; SRCTAG=$TAG; if [ "$MODE" = branch ]; then [ -n "$BRANCHTAG" ] || { echo "$M has no version 3 run to branch from (the 3 c32 lagged members have, c32s4Lt1 at a tau change; the c64 lagged members launch fresh)"; exit 2; }; SRCOUT=$BRANCHOUT; SRCTAG=$BRANCHTAG; fi   # v11: the snapshot's source: the member's own run, or the version 3 run a branch starts from'''), _s(r'''
BRANCHOUT=$SRCOUT; BRANCHTAG=$SRCTAG; SRCOUT=$OUT; SRCTAG=$TAG; if [ "$MODE" = branch ]; then [ -n "$BRANCHTAG" ] || { echo "$M has no source run to branch from (c32s4L, c32s4offL and c32s4Lt1 branch from the version 3 runs, c32s4Lt1 at a tau change; c32s4Lsec from c32s4L's run and c64s8Lrec from c64s8L's at a transport change; the other c64 lagged members launch fresh)"; exit 2; }; SRCOUT=$BRANCHOUT; SRCTAG=$BRANCHTAG; fi   # v11: the snapshot's source: the member's own run, or the source run a branch starts from   # v12: the transport branches''')))

# 6. COMMON
R.append(("cut-transport in COMMON", _s(r'''
--s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --reference-solve $RSOLVE"   # v11: implicit on the version 3 members (version 3 bitwise), lagged on the lagged members and on a gate or segment given the word'''), _s(r'''
--s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --reference-solve $RSOLVE --cut-transport $CUT"   # v11: implicit on the version 3 members (version 3 bitwise), lagged on the lagged members and on a gate or segment given the word   # v12: --cut-transport per member (record on the version 3 members, sector on the L members, the override word)''')))

# 7. write_label: the sentences
R.append(("label sentences", _s(r'''
  LAGSENT="engine_twofield_v8.py (the ruling's engine_twofield_v4; busi/code/compC) with --reference-solve lagged: the reference is transported along the accepted path once per record and held through the record's solve (busi/notes/C64_fix_ruling_v4.md section I.2: at record n the Newton solve finds x_n with the sources Tr[(C_new(x) - P_-(n-1)) rho_op] for every trial x, the reference held; after acceptance each field's reference is transported once from h(x_(n-1)) to h(x_n) along the straight path by the certified walk, the mu S walk on the light and Weyl on the heavy, with the convention as the fallback, exactly as version 3 transports it at its accepted iterate, and stored as P_-(n) with its route, pockets, inversions and weights; the exact J.v omits the reference's derivative, the reference being constant within the record, and is exact in the covariance step and the field solve, the reference held; the lag residual ||G(x_n; P_-(n)) - x_n|| and the consistent collar charge E_cons print per record beside the fields' E; it is the natural discretization: the covariance C is already a state advanced along the accepted path, and the reference becomes one too)"
  RUNSENT="the run to T/r_s $TEND: the 16 r_s collar carries chain-independent radiation from about 39 to 56 and the 8 r_s collar from about 28, the 24 r_s collector radius has no chain-independent window on this chain"
  [ "$TEND" = 56 ] || RUNSENT="the run to T/r_s $TEND, the step control of the formation, the closings and the bounce at the production size (its 2-record comparison with the tau 2 member at matched clocks prints with the readout)"'''), _s(r'''
  LAGSENT="engine_twofield_v9.py (the ruling's engine_twofield_v4 lagged, version 8 with the sector walk of busi/notes/C64_transport_ruling_v1.md section 4; busi/code/compC) with --reference-solve lagged --cut-transport $CUT: the reference is transported along the accepted path once per record and held through the record's solve (busi/notes/C64_fix_ruling_v4.md section I.2: at record n the Newton solve finds x_n with the sources Tr[(C_new(x) - P_-(n-1)) rho_op] for every trial x, the reference held; after acceptance each field's reference is transported once from h(x_(n-1)) to h(x_n) along the straight path by the certified walk, the mu S walk on the light and Weyl on the heavy, exactly as version 3 transports it at its accepted iterate, and stored as P_-(n) with its route, pockets, inversions and weights; the exact J.v omits the reference's derivative, the reference being constant within the record, and is exact in the covariance step and the field solve, the reference held; the lag residual ||G(x_n; P_-(n)) - x_n|| and the consistent collar charge E_cons print per record beside the fields' E; it is the natural discretization: the covariance C is already a state advanced along the accepted path, and the reference becomes one too)"   # v12: engine v9, the transport named
  SECTSENT="where its occupation cut has closed, the occupied subspace is continued piecewise along the path with the lane's declared maximum-overlap convention applied to each crossing at the scale of the crossing and the walk certified at every boundary of the occupied set (the sector walk); the record-scale convention remains the fallback and is flagged (busi/notes/C64_transport_ruling_v1.md 4.3)"   # v12: the transport ruling's 4.3 sentence
  [ "$CUT" = sector ] || SECTSENT="where its occupation cut has closed, the reference is carried by the lane's declared maximum-overlap convention applied to the whole record (--cut-transport record: engine_twofield_v8's transport, bitwise; the transport ruling's 5.1 approach run or 5.9 record-scale branch, not a production transport)"   # v12
  OVERSENT=""; [ -n "${TEND_OVERRIDE:-}" ] && OVERSENT="; --T-over-rs $TEND by TEND_OVERRIDE (the transport ruling 5.1(b): the approach to the clock, to be continued by a resume on --cut-transport sector after G4s8)"   # v12
  RUNSENT="the run to T/r_s $TEND: the 16 r_s collar carries chain-independent radiation from about 39 to 56 and the 8 r_s collar from about 28, the 24 r_s collector radius has no chain-independent window on this chain"
  [ "$TEND" = 56 ] || RUNSENT="the run to T/r_s $TEND, the step control of the formation, the closings and the bounce at the production size (its 2-record comparison with the tau 2 member at matched clocks prints with the readout)"
  [ -n "${TEND_OVERRIDE:-}" ] && RUNSENT="the run to T/r_s $TEND (TEND_OVERRIDE $TEND_OVERRIDE: the clock of this launch, not the production's 56)"   # v12''')))
R.append(("label c64 L", _s(r'''
tau $TAU (1/$STEP r_s per record); $LAGSENT; the engine's head note's "run's end at T/rs 40" is the r_s/a 32 chain's sentence and does not apply, the numbers beside it do; $START; tag $TAG.'''), _s(r'''
tau $TAU (1/$STEP r_s per record); $LAGSENT; $SECTSENT$OVERSENT; the engine's head note's "run's end at T/rs 40" is the r_s/a 32 chain's sentence and does not apply, the numbers beside it do; $START; tag $TAG.''')))
R.append(("label c32 L", _s(r'''
light coupling $COUP; $LAGSENT; $START; its records compare with the stored implicit run at the matched clocks k/16 (D_scheme) and the tau 2 member with the tau 1 member (D_step) by the ruling's reading rule; tag $TAG.'''), _s(r'''
light coupling $COUP; $LAGSENT; $SECTSENT$OVERSENT; $START; its records compare with the stored implicit run at the matched clocks k/16 (D_scheme) and the tau 2 member with the tau 1 member (D_step) by the ruling's reading rule (c32s4Lsec: the r_s/a 32 transport control of the transport ruling's 5.6, D against c32s4L at matched clocks); tag $TAG.''')))
R.append(("label v3 member", _s(r'''
sigma/a $ELL (evolved) prepared at ell 4; light coupling $COUP; tau 2 (1/32 r_s per record); engine_twofield_v3.py unchanged (its head note's "run's end at T/rs 40" is the r_s/a 32 chain's sentence and does not apply; the numbers beside it do); tag $TAG.
EOF
  fi
}'''), _s(r'''
sigma/a $ELL (evolved) prepared at ell 4; light coupling $COUP; tau 2 (1/32 r_s per record); engine_twofield_v9.py --reference-solve implicit --cut-transport record (version 3 bitwise: the fix ruling's G0 and G1, the transport ruling's G0'; its head note's "run's end at T/rs 40" is the r_s/a 32 chain's sentence and does not apply; the numbers beside it do); tag $TAG.
EOF
  fi
}
transport_rewrite() {   # v12: the resume, branch, segment or gate at a transport change (the transport ruling 4.5): the snapshot copy's args.cut_transport (record when the run's json carries none) rewritten to this launch's as R4 rewrites tau, the old value kept as args.cut_transport_before, the archive untouched; the engine's resume check (light_resume_v36) compares the argument
  SRCCUT=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['args'].get('cut_transport','record'))" $SNAP/collapse_${SRCTAG}.json) || exit 4
  CUTNOTE=""
  if [ "$SRCCUT" != "$CUT" ]; then
    python -c "import json,sys; p=sys.argv[1]; J=json.load(open(p)); J['args']['cut_transport_before']=J['args'].get('cut_transport','record'); J['args']['cut_transport']=sys.argv[2]; json.dump(J, open(p,'w'), default=float)" $SNAP/collapse_${SRCTAG}.json "$CUT" || { echo "the snapshot's args.cut_transport could not be rewritten"; exit 4; }
    CUTNOTE=" at a transport change (the run's --cut-transport $SRCCUT to this launch's $CUT from the slice T/rs $TS; the snapshot copy's args.cut_transport rewritten, the old value kept as args.cut_transport_before, the archive untouched)"
    st "$MODE at a transport change: $SRCCUT to $CUT from T/rs $TS; the snapshot copy's args.cut_transport rewritten, the old value kept as args.cut_transport_before"
  fi
}''')))

# 8. the fresh launch
R.append(("fresh launch", _s(r'''
  st "launch $gpuinfo"
  START="from T = 0"; write_label   # v11: the LABEL (version 9's text on the version 3 r_s/a 64 members, the fix ruling's sentence on the lagged members)
  nice -n 10 python $B/code/engine_twofield_v8.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/$OUT --tag $TAG > $B/logs/C${RS}_$M.log 2>&1 &   # v11: engine v8, the member's out'''), _s(r'''
  st "launch (--cut-transport $CUT, --T-over-rs $TEND${TEND_OVERRIDE:+ by TEND_OVERRIDE}) $gpuinfo"   # v12
  START="from T = 0"; write_label   # v11: the LABEL (version 9's text on the version 3 r_s/a 64 members, the fix ruling's sentence on the lagged members)   # v12: the transport ruling's 4.3 sentence on a sector launch
  nice -n 10 python $B/code/engine_twofield_v9.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/$OUT --tag $TAG > $B/logs/C${RS}_$M.log 2>&1 &   # v11: the member's out   # v12: engine v9''')))

# 9. the branch mode: the transport rewrite, the START text, the engine
R.append(("branch", _s(r'''
  START="started by the branch mode from the slice $SLICE (T/rs $TS) of the version 3 run $SRCTAG, whose records to that clock are the implicit scheme's (the head's resume_slice_v34 names the slice and the run)$TAUNOTE"
  write_label
  st "branch from $SLICE (T/rs $TS) of $SRCTAG, snapshot $SNAP; $gpuinfo"
  nice -n 10 python $B/code/engine_twofield_v8.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/$OUT --tag $TAG ''') + '\\', _s(r'''
  transport_rewrite   # v12: the branch at a transport change (c32s4Lsec from c32s4L's record-scale run on the sector walk; c64s8Lrec from c64s8L's run on the record transport)
  START="started by the branch mode from the slice $SLICE (T/rs $TS) of the run $SRCTAG (version 3's implicit records, or the lagged run's own, to that clock; the head's resume_slice_v34 names the slice and the run)$TAUNOTE$CUTNOTE"   # v12
  write_label
  st "branch from $SLICE (T/rs $TS) of $SRCTAG with --cut-transport $CUT, snapshot $SNAP; $gpuinfo"   # v12
  nice -n 10 python $B/code/engine_twofield_v9.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/$OUT --tag $TAG ''') + '\\'))

# 10. the resume mode: the transport rewrite, the LABEL appended, the engine
R.append(("resume", _s(r'''
  st "resume from $SLICE (T/rs $TS), snapshot $SNAP; $gpuinfo"
  nice -n 10 python $B/code/engine_twofield_v8.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/$OUT --tag $TAG ''') + '\\', _s(r'''
  transport_rewrite   # v12: the resume at a transport change (the sigma/a 8 member's continuation from its 13.0 slice on the sector walk, the transport ruling 5.4(a))
  [ -n "$CUTNOTE" ] && echo "Continued from the slice $SLICE (T/rs $TS) by the resume mode$CUTNOTE; the records to T/rs $TS are the record transport's, the records after it the sector walk's; $(date -u +%Y-%m-%dT%H:%M:%SZ)." >> $B/$OUT/LABEL_$TAG.txt   # v12: the LABEL says so with the change clock
  st "resume from $SLICE (T/rs $TS) with --cut-transport $CUT, snapshot $SNAP; $gpuinfo"   # v12
  nice -n 10 python $B/code/engine_twofield_v9.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/$OUT --tag $TAG ''') + '\\'))

# 11. the segment mode: the directory, the transport rewrite, the engine
R.append(("segment", _s(r'''
  G=$B/gate/segment_${M}_${RSOLVE}_$stamp; mkdir -p $G/states
  NSEG=$(python -c "import sys; print(int(round((float(sys.argv[2]) - float(sys.argv[1])) * float(sys.argv[3]) / float(sys.argv[4]))))" "$TS" "$SEGEND" "$RS" "$TAU")
  st "segment from $SLICE (T/rs $TS to $SEGEND, $NSEG records at tau/rs = $TAU/$RS) in mode $RSOLVE, snapshot $SNAP, out $G; $gpuinfo"
  nice -n 10 python $B/code/engine_twofield_v8.py $COMMON --slices-c128 "" --slices-light-c128 "" --ell $ELL --slice-dir $G/states --gpu 0 --out $G --tag $TAG ''') + '\\', _s(r'''
  G=$B/gate/segment_${M}_${RSOLVE}_${CUT}_$stamp; mkdir -p $G/states   # v12: the directory carries the transport word
  NSEG=$(python -c "import sys; print(int(round((float(sys.argv[2]) - float(sys.argv[1])) * float(sys.argv[3]) / float(sys.argv[4]))))" "$TS" "$SEGEND" "$RS" "$TAU")
  transport_rewrite   # v12: the gates G4, G4c, G4s8 run the version 3 members' slices on the sector walk: the snapshot copy's transport rewritten for the engine's resume check
  st "segment from $SLICE (T/rs $TS to $SEGEND, $NSEG records at tau/rs = $TAU/$RS) in mode $RSOLVE with --cut-transport $CUT, snapshot $SNAP, out $G; $gpuinfo"   # v12
  nice -n 10 python $B/code/engine_twofield_v9.py $COMMON --slices-c128 "" --slices-light-c128 "" --ell $ELL --slice-dir $G/states --gpu 0 --out $G --tag $TAG ''') + '\\'))

# 12. the gate mode: the directory, the transport rewrite, the engine
R.append(("gate", _s(r'''
G=$B/gate/${stamp}_${M}_$RSOLVE; mkdir -p $G/states   # v11: the gate's directory names the reference solve
TE=$(python -c "import sys; print(repr(float(sys.argv[1]) + int(sys.argv[2]) * float(sys.argv[4]) / float(sys.argv[3])))" "$TS" "$NREC" "$RS" "$TAU")   # v11: the record step tau/rs
st "resume gate from $SLICE (T/rs $TS to $TE) in mode $RSOLVE, snapshot $SNAP, out $G; $gpuinfo"
nice -n 10 python $B/code/engine_twofield_v8.py $COMMON --slices-c128 "" --slices-light-c128 "" --ell $ELL --slice-dir $G/states --gpu 0 --out $G --tag $TAG ''') + '\\', _s(r'''
G=$B/gate/${stamp}_${M}_${RSOLVE}_$CUT; mkdir -p $G/states   # v11: the gate's directory names the reference solve   # v12: and the transport
TE=$(python -c "import sys; print(repr(float(sys.argv[1]) + int(sys.argv[2]) * float(sys.argv[4]) / float(sys.argv[3])))" "$TS" "$NREC" "$RS" "$TAU")   # v11: the record step tau/rs
transport_rewrite   # v12
st "resume gate from $SLICE (T/rs $TS to $TE) in mode $RSOLVE with --cut-transport $CUT, snapshot $SNAP, out $G; $gpuinfo"   # v12
nice -n 10 python $B/code/engine_twofield_v9.py $COMMON --slices-c128 "" --slices-light-c128 "" --ell $ELL --slice-dir $G/states --gpu 0 --out $G --tag $TAG ''') + '\\'))


def build(src):
    out = src
    log = []
    for name, old, new in R:
        n = src.count(old)
        if n != 1:
            raise SystemExit("build: the old string of %r occurs %d times in version 11, expected exactly once; the version 11 text differs from input 09 at that place; nothing written" % (name, n))
        if out.count(old) != 1:
            raise SystemExit("build: the old string of %r does not occur exactly once at its turn; nothing written" % name)
        out = out.replace(old, new, 1)
        log.append((name, src[:src.index(old)].count("\n") + 1))
    return out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v11", required=True, help="pod_launch_C_v11.sh (input 09)")
    ap.add_argument("--out", required=True, help="pod_launch_C_v12.sh to write")
    a = ap.parse_args()
    src = open(a.v11, encoding="utf-8").read()
    if "# pod_launch_C_v11.sh (busi lane): the fix ruling's pod_launch_C_v10" not in src or "--cut-transport" in src:
        raise SystemExit("build: %s is not pod_launch_C_v11.sh (or already carries the transport word)" % a.v11)
    out, log = build(src)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(out)
    if shutil.which("bash"):
        rc = subprocess.call(["bash", "-n", a.out])
        if rc != 0:
            raise SystemExit("build: bash -n rejects %s (exit %d)" % (a.out, rc))
        syntax = "bash -n passed"
    else:
        syntax = "bash not found here: run bash -n on the pod before use"
    print("built %s from %s: %d replacements, each old string once (%s); %d -> %d lines; %s" % (
        a.out, a.v11, len(log), ", ".join("%s at v11 line %d" % q for q in log), len(src.splitlines()), len(out.splitlines()), syntax))


if __name__ == "__main__":
    main()
