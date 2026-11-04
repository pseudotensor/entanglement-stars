#!/bin/bash
# pod_launch_C_v11.sh (busi lane): the fix ruling's pod_launch_C_v10 (busi/notes/C64_fix_ruling_v4.md, inbox busi #193 item 2), filed as version
# 11 because pod_launch_C_v10.sh exists in this lane (version 8 plus s4offc50); built over version 9 by build_pod_launch_C_v11.py (each old
# string once, else it aborts). Version 11: (i) the engine is engine_twofield_v8.py, the ruling's engine_twofield_v4, for every member; a version
# 3 member's fresh, resume, gate and segment modes pass --reference-solve implicit (version 3 bitwise; the ruling's G1) unless the word "lagged"
# follows the mode's arguments; (ii) the lagged members, every one with --reference-solve lagged, out and LABEL under results/C64v4L or
# results/C32v4L, the fix ruling's I.2 sentence in the LABEL: c64s4L, c64s4offL, c64s8L, c64s8offL (RS 64, tau 2, TEND 56, tags
# C64_s4_ell4_tau2_twofield_v4L, ..._off, C64_s8_ell8_tau2_twofield_v4L, ..._off), c64s4Lt1 (tau 1, TEND 16, C64_s4_ell4_tau1_twofield_v4L),
# c32s4L and c32s4offL (the r_s/a 32 sigma/a 4 member's and control's stored argument sets, version 9's s4 and s4off, with --T-over-rs 56; tags
# C32_s4_ell4_tau2_twofield_v4L and ..._off), c32s4Lt1 (tau 1, TEND 24, C32_s4_ell4_tau1_twofield_v4L); their orbitals slices at every integer
# clock 0 to TEND and light-only slices at the half clocks 10.5 to TEND - 0.5; (iii) the gate mode's end clock and the segment mode's record
# count from the record step tau/rs; (iv) 2 modes added:
#   pod_launch_C_v11.sh MEMBER segment SLICE END [lagged|implicit] [BASE]   the restart from SLICE for the records to T/rs END into
#                                                      gate/segment_<member>_<rsolve>_<stamp>/, no slices written, no compare (the ruling's G3);
#   pod_launch_C_v11.sh MEMBER branch SLICE [BASE]     a lagged member started from its version 3 run's slice (c32s4L and c32s4offL from the
#                                                      r_s/a 32 member's and control's slices, the ruling's G2): the version 3 run's files are
#                                                      snapshotted and the lagged run is written under the member's own tag and out; at a tau change
#                                                      (c32s4Lt1 from the tau 2 member's slice) the snapshot copy's args.tau is rewritten to the
#                                                      member's for the engine's resume check and the run ends by the clock TEND (the slice is a
#                                                      complete state, propagated by the member's tau; the LABEL and status.txt say so);
#   pod_launch_C_v11.sh MEMBER gate SLICE [N] [lagged|implicit] [BASE]      version 9's gate with the record step tau/rs;
# (v) the memory guards of version 9 unchanged (76,600 MiB for a fresh, resumed, branched or segment run of an r_s/a 64 member, 42,000 for its
# gate; the r_s/a 32 members unguarded as before, the c32 lagged members hold the same share). The version 3 members' arguments are version 9's,
# unchanged. Version 9's header follows.
# pod_launch_C_v9.sh (busi lane): version 8 plus the 4 members of Computation C's averaging control at r_s/a 64 (busi/specs/C64_spec_v1.md;
# the owner's funding, inbox #140; REVIEWA_REORIENT_4.md item 11, C5): c64s4 and c64s4off (sigma/a 4, coupling on and off) and c64s8 and
# c64s8off (sigma/a 8), each on 2048 sites at --rs 64 (the chain's end at 32 r_s), tau 2 (1/32 r_s per record), prepared at prep-ell 4 and
# evolved at ell 4 or 8, the Newton loop and the light arguments of the r_s/a 32 sigma/a 4 member unchanged, to T/r_s 56 (1793 records; the
# chain end's return into 16 r_s by the vacuum form is near 56.6, the engine's print on the T = 0 fields the record): complex128 orbitals
# slices at every 1 r_s 0 to 56 and light-only slices at the half clocks 10.5 to 55.5; out results/C64, tags C64_*. Version 9 also (i)
# computes the gate mode's end clock with the record step tau/rs from RS (1/32 at r_s/a 64; version 8's 2/32 was the r_s/a 32 step), (ii)
# refuses a fresh or resumed launch of an r_s/a 64 member below 76600 MiB free on the card (the measured 66.8 GB share, read as 68403 MiB,
# plus the 8192 MiB guard; a gate below 42000 MiB), so that no live run is squeezed, and (iii) writes results/C64/LABEL_<tag>.txt at a fresh
# launch (the ruling's label, which the engine's head note cannot carry). The r_s/a 32 members are version 8's, unchanged. Version 8's
# header follows.
# pod_launch_C_v8.sh (busi lane): version 7 plus the continuations s4c80 and s4offc80 (reorientation 4, REVIEWA_REORIENT_4.md, the busi entry and
# MOVES "pod 1"): the sigma/a 4 member and its coupling-off control continued from their T/r_s 40 slices to 80 on the identical chain, solver
# and arguments, because the formation's light reaches the 16 r_s collar only from about T/r_s 39 and 24 r_s from about 49 (the chain end's
# return into 24 r_s is near 114). Each continuation is the member's own run (same tag, same run files, the resume mode's snapshot), with only
# --T-over-rs 80 and the slice clocks changed: complex128 orbitals slices at every 1 r_s from 41 to 80 (the resumability rule, turn 56
# 48.4 (3)) and light-only slices at the half clocks 40.5 to 79.5. Use the resume and gate modes only (a fresh launch of a c80 member is
# refused). The members s4, s2, s2off and s4off are version 7's, unchanged. Version 7's header follows.
# pod_launch_C_v7.sh (busi lane): version 6 plus the member s4off, the coupling-off control at sigma/a 4 on the pathfinder's chain that the
# sigma/a 4 reading ranks first (busi/notes/C_s4_readouts_reading_v1.1.md (d) 1): without it readouts 3 and 5 cannot be printed as the
# coupling's. --light-coupling off, --nk-jvp exact, the pathfinder's arguments, tag ..._off on every output; not a C member, no item 5 readout
# taken from it; its P2 readout prints and never stops. Version 6's header follows.
# pod_launch_C_v6.sh (busi lane): version 5 plus the member s2off, the labelled coupling-off sigma/a 2 control the sigma/a 2 judgment admits
# (busi/notes/C_s2_start_judgment_v1.md sec. 3: the spec's sec. 6 solver-matched first-order control, tag ..._off, not a Computation C member, no
# back-reaction readout taken from it). --light-coupling off with --nk-jvp exact (gate G2's form, bitwise against v4's option); everything else
# the pathfinder's. Its T = 0 record is written UNCONVERGED below the end rule (kept 1.88e-04, as the production member2's 1.86e-04), and the
# history continues from 0.0625 as member2's does. Version 5's header follows.
# pod_launch_C_v5.sh (busi lane): version 4 with the sigma/a 2 member on the Newton loop's finite-difference J.v (no --nk-jvp exact) and its own
# tag C32_s2_ell2_tau2_twofield_v3_fd. The first sigma/a 2 launch (version 4, --nk-jvp exact) ended at its second record (T/r_s 0.0625) by the
# engine's end rule: the direct solve stalled at Newton step 2 at T = 0 (kept 2.73e-04) and stage 2 of the source continuation at 0.0625 (kept
# 2.14e-03 against 1e-03). The production's r_s/a 32 sigma/a 2 member (LADrs32 member2_snapshot_2100, engine_collapse_v32) and this lane's gate C
# for sigma/a 2 (which rebuilt the stored arguments, without --nk-jvp) ran the finite-difference J.v. The sigma/a 4 member keeps --nk-jvp exact.
# Version 4's header follows.
# pod_launch_C_v4.sh (busi lane): version 3 with the complex128 orbitals slices at every 1 r_s, 0 to 40 (version 3: 0, 4, 8, 12, 13 to 20, 22,
# 24, 28, 32, 36, 40). Solve lead turn 56 sec. 48.4 (3): a launcher counts as resumable when it writes a restart state at least every 1 r_s,
# mirrors it off the pod, and its runner has reproduced 2 records within the floor from such a state (the resume gate of
# chain_C_s4_gate_s2_pod3 runs just before). The light-only slices keep version 3's clocks; the engine writes none at an orbitals clock.
# Version 3's header follows.
# Computation C's pathfinder, one member on an H100 pod, with the restart path (the resumability audit, inbox #86). pod_launch_C_v2.sh plus 2 modes:
#   pod_launch_C_v3.sh s4|s2 [BASE]                    a launch from T = 0, exactly as version 2;
#   pod_launch_C_v3.sh s4|s2 resume SLICE [BASE]       the restart: the member's run files are copied to BASE/snap/<stamp>/ (the engine reads the
#                                                      stored rows from that copy, never from files it is rewriting), then the engine continues from
#                                                      the complex128 orbitals slice SLICE into the same out dir and tag (version 34 (a) with
#                                                      light_resume_v36: the light state comes from the same slice). Refused while the member's
#                                                      recorded PID is alive;
#   pod_launch_C_v3.sh s4|s2 gate SLICE [N] [BASE]     the resume gate: the same restart for N records (default 2) into BASE/gate/<stamp>/ with no
#                                                      slices written, then compare_resume_gate_v1.py against the snapshot: BITWISE, WITHIN-FLOOR or
#                                                      FAIL. Safe beside a live member (it reads a snapshot) if the device has room (about 33 GB).
# The slice must be on the pod: c_mirror_v3 keeps the latest 2 orbitals slices there and the older ones on the box (${QI2_ROOT}/busi_bulk) and the
# owner's dataset; a restart from an older one needs it copied back first (scp from the box). Restart loss: at most the records after SLICE.
M=${1:?member s4, s2, s2off, s4off, s4c80, s4offc80, c64s4, c64s4off, c64s8, c64s8off, c64s4L, c64s4offL, c64s8L, c64s8offL, c64s4Lt1, c32s4L, c32s4offL or c32s4Lt1}; MODE=fresh   # v11
case ${2:-} in resume|gate|branch) MODE=$2; SLICE=${3:?slice}; shift 3;; segment) MODE=segment; SLICE=${3:?slice}; SEGEND=${4:?the end clock T/r_s of the segment}; shift 4;; *) shift 1;; esac   # v11: the segment and branch modes
NREC=2; if [ "$MODE" = gate ] && [[ ${1:-} =~ ^[0-9]+$ ]]; then NREC=$1; shift; fi
RSOLVE=; case ${1:-} in implicit|lagged) RSOLVE=$1; shift;; esac   # v11: the reference solve of a version 3 member's gate or segment (implicit when absent); a lagged member always runs lagged
B=${1:-/root/busi_compC}
COUP=on; TEND=40; RS=32; TAU=2; LAG=0; SRCOUT=; SRCTAG=; case $M in c64s4) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=56;; c64s4off) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=56;; c64s8) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=56;; c64s8off) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=56;; s4) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3; JVP="--nk-jvp exact";; s2) ELL=2; TAG=C32_s2_ell2_tau2_twofield_v3_fd; JVP="";; s2off) ELL=2; TAG=C32_s2_ell2_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off;; s4off) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off;; s4c80) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3; JVP="--nk-jvp exact"; TEND=80;; s4offc80) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v3_off; JVP="--nk-jvp exact"; COUP=off; TEND=80;; c64s4L) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v4L; JVP="--nk-jvp exact"; TEND=56; LAG=1;; c64s4offL) RS=64; ELL=4; TAG=C64_s4_ell4_tau2_twofield_v4L_off; JVP="--nk-jvp exact"; COUP=off; TEND=56; LAG=1;; c64s8L) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v4L; JVP="--nk-jvp exact"; TEND=56; LAG=1;; c64s8offL) RS=64; ELL=8; TAG=C64_s8_ell8_tau2_twofield_v4L_off; JVP="--nk-jvp exact"; COUP=off; TEND=56; LAG=1;; c64s4Lt1) RS=64; ELL=4; TAG=C64_s4_ell4_tau1_twofield_v4L; JVP="--nk-jvp exact"; TEND=16; TAU=1; LAG=1;; c32s4L) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v4L; JVP="--nk-jvp exact"; TEND=56; LAG=1; SRCOUT=results/C32; SRCTAG=C32_s4_ell4_tau2_twofield_v3;; c32s4offL) ELL=4; TAG=C32_s4_ell4_tau2_twofield_v4L_off; JVP="--nk-jvp exact"; COUP=off; TEND=56; LAG=1; SRCOUT=results/C32; SRCTAG=C32_s4_ell4_tau2_twofield_v3_off;; c32s4Lt1) ELL=4; TAG=C32_s4_ell4_tau1_twofield_v4L; JVP="--nk-jvp exact"; TEND=24; TAU=1; LAG=1; SRCOUT=results/C32; SRCTAG=C32_s4_ell4_tau2_twofield_v3;; *) echo "member s4, s2, s2off, s4off, s4c80, s4offc80, c64s4, c64s4off, c64s8, c64s8off, c64s4L, c64s4offL, c64s8L, c64s8offL, c64s4Lt1, c32s4L, c32s4offL or c32s4Lt1"; exit 2;; esac   # v9   # v11: TAU and LAG per member, the version 3 run a branch starts from, the 8 lagged members
OUT=results/C$RS; [ "$LAG" = 1 ] && OUT=results/C${RS}v4L   # v11: the lagged members' out directory (the ruling's results/C64v4L, results/C32v4L)
TEND=${TEND_OVERRIDE:-$TEND}   # busi 2026-10-18 (builder-z 9ae0d5c4; C64_transport_ruling_v1 5.1 (b)): the one-line TEND_OVERRIDE edit to v11, marked and recorded; unset it changes nothing
if [ "$LAG" = 1 ]; then [ "$RSOLVE" = implicit ] && { echo "$M is a lagged member: its gates and segments run lagged (the ruling's item 5)"; exit 2; }; RSOLVE=lagged; fi; RSOLVE=${RSOLVE:-implicit}   # v11
if [ "$LAG" = 0 ] && [ "$RSOLVE" = lagged ] && [ "$MODE" != gate ] && [ "$MODE" != segment ]; then echo "$M is a version 3 member: the word lagged applies to its gate and segment modes only (a fresh or resumed lagged run under a version 3 tag and LABEL would write the lagged scheme into the implicit history's files; the ruling: no version 3 run continues); use a lagged member"; exit 2; fi   # v11 (red team R3)
BRANCHOUT=$SRCOUT; BRANCHTAG=$SRCTAG; SRCOUT=$OUT; SRCTAG=$TAG; if [ "$MODE" = branch ]; then [ -n "$BRANCHTAG" ] || { echo "$M has no version 3 run to branch from (the 3 c32 lagged members have, c32s4Lt1 at a tau change; the c64 lagged members launch fresh)"; exit 2; }; SRCOUT=$BRANCHOUT; SRCTAG=$BRANCHTAG; fi   # v11: the snapshot's source: the member's own run, or the version 3 run a branch starts from
cd $B || exit 1; mkdir -p $OUT states logs   # v11: the member's out directory
export BUSI_ENGINE_DIR=$B/engine BUSI_PAGE_DIR=$B/engine PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SL4="--slices-c128 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40 --slices-light-c128 1,2,3,5,6,7,9,10,10.5,11,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21,21.5,22.5,23,23.5,25,26,27,29,30,31,33,34,35,37,38,39"
if [ "$TEND" = 80 ]; then   # v8: the continuation's slice clocks
  [ "$MODE" = fresh ] && { echo "$M is a continuation: use resume or gate"; exit 2; }
  SL4="--slices-c128 $(seq -s, 41 80) --slices-light-c128 $(seq -s, 40.5 1 79.5)"
fi
if [ "$RS" = 64 ]; then   # v9: the r_s/a 64 members' slice clocks (C64_spec_v1.md sec. 1.5): orbitals every 1 r_s 0 to 56 (restarts), light-only at the half clocks 10.5 to 55.5
  SL4="--slices-c128 $(seq -s, 0 56) --slices-light-c128 $(seq -s, 10.5 1 55.5)"
fi
if [ "$LAG" = 1 ]; then   # v11: the lagged members' slice clocks: orbitals (both fields, complex128) at every integer clock 0 to TEND (the restart state every 1 r_s; c_mirror keeps the latest 2 on the pod), light-only at the half clocks 10.5 to TEND - 0.5
  SL4="--slices-c128 $(seq -s, 0 $TEND) --slices-light-c128 $(seq -s, 10.5 1 $((TEND - 1)).5)"
fi
COMMON="--m 0.2 --g 0.25 --rs $RS --R0-over-rs 4 --reading A --nsites 2048 --prep-ell 4 --kappa-sea 10 --tau $TAU --T-over-rs $TEND --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 $JVP --delta-rtol-every 10 --commutator-every 1 --streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --light-kappas 1,2 --light-guard-kappas 3 --light-coupling $COUP --light-transport banded --light-d-min 1e-7 --s11-collars 8,16,24 --s11-stop 1 --field-kernel-cache --v36-fixes --reference-solve $RSOLVE"   # v11: implicit on the version 3 members (version 3 bitwise), lagged on the lagged members and on a gate or segment given the word
stamp=$(date -u +%Y%m%dT%H%M%SZ)
if [ "$RS" = 64 ]; then   # v9: an r_s/a 64 member holds the measured r_s/a 32 share (66.8 GB, the same arrays at 2048 sites); nothing fits beside it under the 8192 MiB guard (C64_spec_v1.md sec. 3)
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); NEED=76600; [ "$MODE" = gate ] && NEED=42000
  [ "$FREE" -ge "$NEED" ] 2>/dev/null || { echo "refused: $FREE MiB free on the card, $NEED MiB needed for $M in mode $MODE"; echo "C${RS}_$M refused ($FREE MiB free, $NEED needed, mode $MODE) $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; exit 6; }
fi
st() { echo "C${RS}_$M $* $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $B/status.txt; }
write_label() {   # v11: the LABEL beside the run files: version 9's text on the version 3 r_s/a 64 members (C64_spec_v1.md sec. 1.6), the fix ruling's I.2 sentence on the lagged members
  STEP=$((RS / TAU)); ROLE=member; [ "$COUP" = off ] && ROLE=control
  LAGSENT="engine_twofield_v8.py (the ruling's engine_twofield_v4; busi/code/compC) with --reference-solve lagged: the reference is transported along the accepted path once per record and held through the record's solve (busi/notes/C64_fix_ruling_v4.md section I.2: at record n the Newton solve finds x_n with the sources Tr[(C_new(x) - P_-(n-1)) rho_op] for every trial x, the reference held; after acceptance each field's reference is transported once from h(x_(n-1)) to h(x_n) along the straight path by the certified walk, the mu S walk on the light and Weyl on the heavy, with the convention as the fallback, exactly as version 3 transports it at its accepted iterate, and stored as P_-(n) with its route, pockets, inversions and weights; the exact J.v omits the reference's derivative, the reference being constant within the record, and is exact in the covariance step and the field solve, the reference held; the lag residual ||G(x_n; P_-(n)) - x_n|| and the consistent collar charge E_cons print per record beside the fields' E; it is the natural discretization: the covariance C is already a state advanced along the accepted path, and the reference becomes one too)"
  RUNSENT="the run to T/r_s $TEND: the 16 r_s collar carries chain-independent radiation from about 39 to 56 and the 8 r_s collar from about 28, the 24 r_s collector radius has no chain-independent window on this chain"
  [ "$TEND" = 56 ] || RUNSENT="the run to T/r_s $TEND, the step control of the formation, the closings and the bounce at the production size (its 2-record comparison with the tau 2 member at matched clocks prints with the readout)"
  if [ "$LAG" = 1 ] && [ "$RS" = 64 ]; then
    cat > $B/$OUT/LABEL_$TAG.txt <<EOF
Computation C, the self-consistent 2-field collapse (D16), the averaging control at r_s/a 64 rerun on the transported-reference scheme (busi/notes/C64_fix_ruling_v4.md section I; busi/specs/C64_spec_v1.md for the chain, the collars and the argument set): the heavy field the production's dust at r_s/a 64 (N_B rs/(2gm) = 640, ma 0.2, g 1/4, the levitating-well preparation at prep-ell 4, R0 4 r_s), the light field the massless 4-component Dirac field of the same operator with multiplicity 16 kappa, kappa 1 and 2 sourcing, kappa 3 the truncation guard; the light reference 1(h_m - mu S < 0) continued by the certified walk (fallback the LZ convention, marked); the chain of 2048 sites, its end at 32 r_s; the chain end's first return (the vacuum form from 2 r_s at T = 0; the engine's print on the T = 0 fields is the record): into 24 r_s near T/r_s 50, into 16 r_s near 57, into 8 r_s near 63; the formation's light reaches 16 r_s about 23 r_s and 24 r_s about 33 r_s after the formation; $RUNSENT; sigma/a $ELL (evolved) prepared at ell 4; light coupling $COUP; tau $TAU (1/$STEP r_s per record); $LAGSENT; the engine's head note's "run's end at T/rs 40" is the r_s/a 32 chain's sentence and does not apply, the numbers beside it do; $START; tag $TAG.
EOF
  elif [ "$LAG" = 1 ]; then
    cat > $B/$OUT/LABEL_$TAG.txt <<EOF
Computation C, the self-consistent 2-field collapse (D16), the r_s/a 32 sigma/a 4 $ROLE rerun on the transported-reference scheme as the scheme control of the r_s/a 64 runs (busi/notes/C64_fix_ruling_v4.md section I.3, G2): the stored argument set of the r_s/a 32 sigma/a 4 $ROLE (pod_launch_C_v9's s4 and s4off; the C spec's pathfinder: --rs 32, 2048 sites, the chain's end at 64 r_s, prepared at ell 4, evolved at ell 4, the Newton loop and the light arguments unchanged) with --T-over-rs $TEND and tau $TAU (1/$STEP r_s per record); light coupling $COUP; $LAGSENT; $START; its records compare with the stored implicit run at the matched clocks k/16 (D_scheme) and the tau 2 member with the tau 1 member (D_step) by the ruling's reading rule; tag $TAG.
EOF
  elif [ "$RS" = 64 ]; then
    cat > $B/results/C64/LABEL_$TAG.txt <<EOF
Computation C, the self-consistent 2-field collapse (D16), the averaging control at r_s/a 64 (busi/specs/C64_spec_v1.md): the heavy field the production's dust at r_s/a 64 (N_B rs/(2gm) = 640, ma 0.2, g 1/4, the levitating-well preparation at prep-ell 4, R0 4 r_s), the light field the massless 4-component Dirac field of the same operator with multiplicity 16 kappa, kappa 1 and 2 sourcing, kappa 3 the truncation guard; the light reference 1(h_m - mu S < 0) continued by the certified walk (fallback the LZ convention, marked); the chain of 2048 sites, its end at 32 r_s; the chain end's first return (the vacuum form from 2 r_s at T = 0; the engine's print on the T = 0 fields is the record): into 24 r_s near T/r_s 50, into 16 r_s near 57, into 8 r_s near 63; the formation's light reaches 16 r_s about 23 r_s and 24 r_s about 33 r_s after the formation; the run to T/r_s 56: the 16 r_s collar carries chain-independent radiation from about 39 to 56 and the 8 r_s collar from about 28, the 24 r_s collector radius has no chain-independent window on this chain; sigma/a $ELL (evolved) prepared at ell 4; light coupling $COUP; tau 2 (1/32 r_s per record); engine_twofield_v3.py unchanged (its head note's "run's end at T/rs 40" is the r_s/a 32 chain's sentence and does not apply; the numbers beside it do); tag $TAG.
EOF
  fi
}
gpuinfo="load $(cut -d' ' -f1 /proc/loadavg) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) free_MiB $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
if [ "$MODE" = fresh ]; then
  st "launch $gpuinfo"
  START="from T = 0"; write_label   # v11: the LABEL (version 9's text on the version 3 r_s/a 64 members, the fix ruling's sentence on the lagged members)
  nice -n 10 python $B/code/engine_twofield_v8.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/$OUT --tag $TAG > $B/logs/C${RS}_$M.log 2>&1 &   # v11: engine v8, the member's out
  PID=$!; echo $PID > $B/C${RS}_$M.pid; st "pid $PID"
  wait $PID; st "exit $?"; exit 0
fi
[ -f "$SLICE" ] || { echo "no slice $SLICE on this pod (copy it back from the box first)"; exit 3; }
TS=$(python -c "import numpy as np,sys; Z=np.load(sys.argv[1]); print(repr(float(Z['T'])/float(Z['rs'])))" "$SLICE") || exit 3
SNAP=$B/snap/${stamp}_$M; mkdir -p $SNAP
for try in 1 2 3; do   # a live member rewrites its files every 5 records: copy until the json and every npz read whole
  cp -p $B/$SRCOUT/collapse_${SRCTAG}.json $B/$SRCOUT/collapse_${SRCTAG}_*.npz $SNAP/ || exit 4   # v11: the source run (the member's own, or the version 3 run a branch starts from)
  python -c "import json,glob,sys,numpy as np; json.load(open(sys.argv[1])); [np.load(f, allow_pickle=True).files for f in glob.glob(sys.argv[2])]" \
    $SNAP/collapse_${SRCTAG}.json "$SNAP/collapse_${SRCTAG}_*.npz" 2>/dev/null && break   # v11
  [ $try = 3 ] && { echo "snapshot unreadable 3 times"; exit 4; }; sleep 20
done
if [ "$MODE" = branch ]; then   # v11: a lagged member started from its version 3 run's slice (the ruling's G2); the source's records to the slice's clock are the implicit scheme's, the head's resume_slice_v34 names them
  if [ -f $B/C${RS}_$M.pid ] && kill -0 "$(cat $B/C${RS}_$M.pid)" 2>/dev/null; then echo "member $M still running (PID $(cat $B/C${RS}_$M.pid)); not branched"; exit 5; fi
  [ -e $B/$OUT/collapse_${TAG}.json ] && { echo "$M already has run files in $OUT: use resume (its own slices) or move them away first"; exit 2; }
  SRCTAU=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['args']['tau'])" $SNAP/collapse_${SRCTAG}.json)
  SEGARG=; TAUNOTE=
  if ! python -c "import sys; sys.exit(0 if float(sys.argv[1]) == float(sys.argv[2]) else 1)" "$SRCTAU" "$TAU"; then   # v11 (red team R4): a tau change at the branch, the ruling's c32s4Lt1 from the tau 2 member's T0012.00 slice: the slice is a complete state at its clock and the engine's resume propagates it by this call's tau (resume_slice_v34 and light_resume_v36 use a.tau), so the only obstacle is resume_slice_v34's argument check against the stored args.tau; it reads the snapshot copy, never the archive, whose args.tau is rewritten here to this member's (the old value kept as args.tau_before_branch); the loop's step counter then starts at the source's row index, so the run's end is set by the clock, --segment-end-T-over-rs TEND (the step-indexed cadences, the saves every 5th loop step, M9 and the Simpson diagnostic, change no record's numbers)
    python -c "import json,sys; p=sys.argv[1]; J=json.load(open(p)); J['args']['tau_before_branch']=J['args']['tau']; J['args']['tau']=float(sys.argv[2]); json.dump(J, open(p,'w'), default=float)" $SNAP/collapse_${SRCTAG}.json "$TAU" || { echo "the snapshot's args.tau could not be rewritten"; exit 4; }
    SEGARG="--segment-end-T-over-rs $TEND"; TAUNOTE=" at tau $TAU (the source ran at tau $SRCTAU; the snapshot copy's args.tau rewritten to $TAU for the engine's resume check, the slice's state at T/rs $TS propagated by tau $TAU; the run ends by the clock $TEND)"
    st "branch at a tau change: the source's tau $SRCTAU, this member's $TAU; the snapshot copy's args.tau rewritten; the end by the clock $TEND"
  fi
  START="started by the branch mode from the slice $SLICE (T/rs $TS) of the version 3 run $SRCTAG, whose records to that clock are the implicit scheme's (the head's resume_slice_v34 names the slice and the run)$TAUNOTE"
  write_label
  st "branch from $SLICE (T/rs $TS) of $SRCTAG, snapshot $SNAP; $gpuinfo"
  nice -n 10 python $B/code/engine_twofield_v8.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/$OUT --tag $TAG \
    --resume-slice "$SLICE" --resume-run $SNAP/$SRCTAG $SEGARG > $B/logs/C${RS}_${M}_branch_$stamp.log 2>&1 &
  PID=$!; echo $PID > $B/C${RS}_$M.pid; st "pid $PID (branched)"
  wait $PID; st "exit $? (branched)"; exit 0
fi
if [ "$MODE" = resume ]; then   # v11: engine v8, the member's out
  if [ -f $B/C${RS}_$M.pid ] && kill -0 "$(cat $B/C${RS}_$M.pid)" 2>/dev/null; then echo "member $M still running (PID $(cat $B/C${RS}_$M.pid)); not restarted"; exit 5; fi
  st "resume from $SLICE (T/rs $TS), snapshot $SNAP; $gpuinfo"
  nice -n 10 python $B/code/engine_twofield_v8.py $COMMON $SL4 --ell $ELL --slice-dir $B/states --gpu 0 --out $B/$OUT --tag $TAG \
    --resume-slice "$SLICE" --resume-run $SNAP/$TAG > $B/logs/C${RS}_${M}_resume_$stamp.log 2>&1 &
  PID=$!; echo $PID > $B/C${RS}_$M.pid; st "pid $PID (resumed)"
  wait $PID; st "exit $? (resumed)"; exit 0
fi
if [ "$MODE" = segment ]; then   # v11: the segment (the ruling's G3): the restart from SLICE for the records to T/rs SEGEND into gate/segment_<member>_<rsolve>_<stamp>/, no slices written, no compare; the record count from the record step tau/rs
  G=$B/gate/segment_${M}_${RSOLVE}_$stamp; mkdir -p $G/states
  NSEG=$(python -c "import sys; print(int(round((float(sys.argv[2]) - float(sys.argv[1])) * float(sys.argv[3]) / float(sys.argv[4]))))" "$TS" "$SEGEND" "$RS" "$TAU")
  st "segment from $SLICE (T/rs $TS to $SEGEND, $NSEG records at tau/rs = $TAU/$RS) in mode $RSOLVE, snapshot $SNAP, out $G; $gpuinfo"
  nice -n 10 python $B/code/engine_twofield_v8.py $COMMON --slices-c128 "" --slices-light-c128 "" --ell $ELL --slice-dir $G/states --gpu 0 --out $G --tag $TAG \
    --resume-slice "$SLICE" --resume-run $SNAP/$SRCTAG --segment-end-T-over-rs $SEGEND > $G/segment_run.log 2>&1
  rc=$?; SUMMARY=$(python -c "import json,sys; J=json.load(open(sys.argv[1])); R=[r for r in J['records'] if r.get('E') is not None]; print(len(R), 'records, the last at T/rs', R[-1]['T_over_rs'], ';', 'end_v32' if 'end_v32' in J else 'no end_v32', ';', 'end_s11' if 'end_s11' in J else 'no end_s11')" $G/collapse_$TAG.json 2>/dev/null)
  st "segment run exit $rc: $SUMMARY"; exit $rc
fi
G=$B/gate/${stamp}_${M}_$RSOLVE; mkdir -p $G/states   # v11: the gate's directory names the reference solve
TE=$(python -c "import sys; print(repr(float(sys.argv[1]) + int(sys.argv[2]) * float(sys.argv[4]) / float(sys.argv[3])))" "$TS" "$NREC" "$RS" "$TAU")   # v11: the record step tau/rs
st "resume gate from $SLICE (T/rs $TS to $TE) in mode $RSOLVE, snapshot $SNAP, out $G; $gpuinfo"
nice -n 10 python $B/code/engine_twofield_v8.py $COMMON --slices-c128 "" --slices-light-c128 "" --ell $ELL --slice-dir $G/states --gpu 0 --out $G --tag $TAG \
  --resume-slice "$SLICE" --resume-run $SNAP/$TAG --segment-end-T-over-rs $TE > $G/gate_run.log 2>&1
rc=$?; st "resume gate run exit $rc"
python $B/code/compare_resume_gate_v1.py --stored-dir $SNAP --gate-dir $G --tag $TAG --slice-T-over-rs $TS --out $G/resume_gate.json > $G/resume_gate.log 2>&1
rc2=$?; st "resume gate compare exit $rc2: $(tail -1 $G/resume_gate.log)"; exit $rc2
