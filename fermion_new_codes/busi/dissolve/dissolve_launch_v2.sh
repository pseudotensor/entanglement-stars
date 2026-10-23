#!/bin/bash
# dissolve_launch_v2.sh (busi, reviewer A red team DISSOLVE_redteam_reviewa_v1, 2026-10-20): dissolve_launch_v1.sh (ebc71570) with the red team's
# fixes; everything not listed here is version 1's, byte for byte in the member lines and the modes.
#   (1) P0 (C6-3, --reference R1) runs WITHOUT --nk-jvp exact --pair-workers 2: the exact J.v (engine_collapse_v36b lines 2765 to 2808)
#       differentiates the reference's density through the negative-energy mask, which is the adiabatic sea's response under R2; under R1 the
#       reference is the propagated sea itself (lines 2065, 2101, 2538, 2736) and the sources do not depend on the trial fields, so that
#       direction is wrong by the sea's screening term. Without the option the Newton direction is version 32's finite-difference Krylov
#       product (nk_loop_v32, line 2651), exact for a field-independent source map (J = -I): one Krylov product, the step lands on the fixed
#       point, about 4 evaluations per record.
#   (2) The slice lists: G saves the episode (1.25, 1.5, 1.75, 1.8125, 2.0, 2.0625, 2.25, 2.3125, 2.4375, 2.5, 2.5625: the formation, the deepest
#       record and +0.25, +0.5, +0.75 r_s from it, the stored 2.0 slice and +0.25, +0.5 from it, the last zero, the first record without one);
#       L2, S1, S05 save 1, 1.75, 1.8125, 2, 2.5, 3, 4, 5, 6; L8 the integers 1 to 6; P0 1.8125, 2.0, 2.5; W 2.3125, 2.5; P40 10, 20, 30, 40.
#   (3) X05 and X20 are kept as members but are NOT in the planned set (the convention's parameter cannot act on these records: the ruling's
#       section 2.4); X20_05 (c_LZ 2 to T/r_s 0.5, 8 records, no slices) is the optional 4-minute bitwise check of that reading against G.
#   (4) main mode refuses when the device has no CUDA (GPU != cpu) or when the run root's filesystem has less than the member's slices plus
#       5 GB free (2.8 GB per 2048-site slice, 11 GB per 4096-site slice), so nothing is launched that cannot finish.
# Usage: dissolve_launch_v2.sh <member> main|print|dry
#   members: G (C6-1), P0 (C6-3), L2 (C6-5), L8 (C6-6), S1 (C6-7), W (C6-8), S05 (C6-9), M64 (C6-11),
#            P40_L2 P40_L8 P40_S1 P40_S05 (C6-10, the continuation of that member from its T/r_s = 6 slice to 40),
#            X05 X20 (C6-4 as the spec wrote it; not planned), X20_05 (the optional check), PREP0 (the box CPU known answer, record 0 only)
# Environment: KIT (default ${QI2_ROOT}/kit_v1), DROOT (default ${QI2_ROOT}/dissolve: runs/, states/, logs/, pids/, gates/),
#   PY (default ${QI2_ROOT}/python: torch 2.8.0 is required, the massless preparation fails on torch 2.4.1, solve STATUS),
#   GPU (default 0; cpu omits --gpu, the box CPU known answer PREP0 only). THREADS (default 8). DEADLINE (seconds; the engine under timeout;
#   unset on the cards, where the solve lane's queue governs). GATE=0 skips the C6-1 PASS-file requirement (dry checks only). TORCH_ANY=1 skips
#   the torch 2.8.0 check (dry only). DISK_ANY=1 skips the free-space check (dry and PREP0 only).
# The engine runs with only the kit's code directory on its path: python -s -B with PYTHONPATH unset, cwd the member's own empty run dir.
# This script never kills anything and never touches a queue file.
set -u
KIT=${KIT:-${QI2_ROOT}/kit_v1}; ROOT=${DROOT:-${QI2_ROOT}/dissolve}; PY=${PY:-${QI2_ROOT}/python}; GPU=${GPU:-0}
C=$KIT/code; ENGINE=$C/engine_collapse_v36b.py; MENG=9794c1b38fe59d19f2bb510673e757f2
BT=pair_rs32_ell4a_v36b
M=${1:-}; MODE=${2:-}
now() { date -u +%FT%TZ; }
die() { echo "$(now) dissolve_launch_v2 $M: $*"; exit 1; }

# the base line (the spec's, token for token; --slices-c128, --gpu, --out, --tag and --slice-dir are set per member below)
BASE=(--m 0.0 --g 0.25 --rs 32.0 --nsites 2048 --ell 4 --tau 2.0 --T-over-rs 40.0 --R0-over-rs 4.0 --kappa-sea 10 --reference R2 --c-LZ 1.0
 --window-max 4 --site-fields averaged --massless-ball --ball-orbitals 1:12 --ball-R0-over-rs 4.0 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25
 --ball-E-target 64.0 --ball-E-tol 1e-10 --ball-tol 1e-10 --ball-passes 200 --ball-k0-max 0.5 --fields-v17 --field-kernel-cache --v36-fixes
 --nk-jvp exact --mix newton --mix-alpha 0.5 --maxit 20 --nk-halvings 6 --nk-inner 15 --nk-krylov 40 --nk-line-search halving --nk-maxiter 30
 --nk-restarts 2 --source-continuation two-stage --end-above 0.001 --stop-E 1e-07 --stop-fields 1e-06 --field-rtol 1e-12 --field-atol 1e-16
 --collar-policy flag --support-rel 1e-14 --commutator-every 1 --delta-rtol-every 10 --m1-simpson-steps auto --half-offset 0 --streams 0
 --pair-workers 2 --reading A --s11-collars 8,16 --s11-stop 1 --weyl-refine-max 10)
ARGS=("${BASE[@]}")
setf() {   # setf FLAG VALUE: replace FLAG's value in ARGS, or append FLAG VALUE if absent
  local f=$1 v=$2 i
  for ((i=0; i<${#ARGS[@]}; i++)); do if [ "${ARGS[$i]}" = "$f" ]; then ARGS[$((i+1))]=$v; return; fi; done
  ARGS+=("$f" "$v"); }
unsetf() {   # unsetf FLAG: remove FLAG and its one value from ARGS (version 2, for P0)
  local f=$1 i; local out=()
  for ((i=0; i<${#ARGS[@]}; i++)); do if [ "${ARGS[$i]}" = "$f" ]; then i=$((i+1)); else out+=("${ARGS[$i]}"); fi; done
  ARGS=("${out[@]}"); }
SL6=1,2,3,4,5,6
SLG=1.25,1.5,1.75,1.8125,2.0,2.0625,2.25,2.3125,2.4375,2.5,2.5625
SLL=1,1.75,1.8125,2,2.5,3,4,5,6
NSLICE_GB=2.8
member() {   # $1 member: sets TAG, SL and the changed flags (the spec's table with the red team's section 2 applied)
  case $1 in
    G)   TAG=${BT}_G;   SL=$SLG; setf --T-over-rs 2.625;;
    P0)  TAG=${BT}_P0;  SL=1.8125,2.0,2.5; setf --reference R1; setf --T-over-rs 3.0; unsetf --nk-jvp; unsetf --pair-workers;;
    X05) TAG=${BT}_X05; SL=""; setf --c-LZ 0.5; setf --T-over-rs 3.0;;
    X20) TAG=${BT}_X20; SL=""; setf --c-LZ 2.0; setf --T-over-rs 3.0;;
    X20_05) TAG=${BT}_X20_05; SL=""; setf --c-LZ 2.0; setf --T-over-rs 0.5;;
    L2)  TAG=${BT}_L2;  SL=$SLL; setf --prep-ell 4; setf --ell 2; setf --T-over-rs 6.0;;
    L8)  TAG=${BT}_L8;  SL=$SL6; setf --prep-ell 4; setf --ell 8; setf --T-over-rs 6.0;;
    S1)  TAG=${BT}_S1;  SL=$SLL; setf --tau 1.0; setf --T-over-rs 6.0;;
    S05) TAG=${BT}_S05; SL=$SLL; setf --tau 0.5; setf --T-over-rs 6.0;;
    W)   TAG=${BT}_W;   SL=2.3125,2.5; setf --T-over-rs 2.625
         RS_SLICE=$ROOT/states/${BT}_G/orbitals_${BT}_G_T0002.00.npz; RS_RUN=$ROOT/runs/${BT}_G/${BT}_G
         setf --resume-slice $RS_SLICE; setf --resume-run $RS_RUN; setf --weyl-certified eigvals; setf --segment-end-T-over-rs 2.625;;
    P40_L2|P40_L8|P40_S1|P40_S05)
         local s=${1#P40_}; member $s; local ST=$TAG; TAG=${ST}_P40; SL=10,20,30,40; setf --T-over-rs 40.0
         RS_SLICE=$ROOT/states/$ST/orbitals_${ST}_T0006.00.npz; RS_RUN=$ROOT/runs/$ST/$ST
         setf --resume-slice $RS_SLICE; setf --resume-run $RS_RUN;;
    PREP0) TAG=${BT}_prep0_cpu; SL=""; setf --T-over-rs 0.0;;   # the preparation's known answer (box CPU, GPU=cpu): record 0 only
    M64) TAG=pair_rs64_ell8a_v36b_M64; SL=$SL6; NSLICE_GB=11; setf --rs 64.0; setf --nsites 4096; setf --ell 8; setf --ball-orbitals 1:24
         setf --ball-E-target 128.0; setf --T-over-rs 6.0;;
    *) die "unknown member '$1' (G P0 L2 L8 S1 W S05 M64 P40_L2 P40_L8 P40_S1 P40_S05 X05 X20 X20_05 PREP0)";;
  esac; }
RS_SLICE=""; RS_RUN=""
member "$M"
OUT=$ROOT/runs/$TAG; STD=$ROOT/states/$TAG; LOG=$ROOT/logs/$TAG.log; PIDF=$ROOT/pids/DISSOLVE_$M.pid
ARGS+=(--slices-c128 "$SL" --slice-dir $STD); [ "$GPU" = cpu ] || ARGS+=(--gpu $GPU); ARGS+=(--out $OUT --tag $TAG)
NSL=0; [ -n "$SL" ] && NSL=$(echo "$SL" | tr ',' '\n' | grep -c .)

checks() {   # the refusals every mode shares
  [ -s $ENGINE ] || die "no engine $ENGINE: refused"
  [ "$(md5sum < $ENGINE | cut -d' ' -f1)" = $MENG ] || die "engine md5 is not $MENG: refused"
  if [ "${TORCH_ANY:-0}" != 1 ]; then
    TV=$($PY -s -c "import torch; print(torch.__version__)" 2>/dev/null) || die "$PY cannot import torch: refused"
    case $TV in 2.8.0*) ;; *) die "torch $TV under $PY; the massless preparation needs torch 2.8.0 (solve STATUS): refused";; esac
  fi
  if [ "$M" != G ] && [ "${GATE:-1}" != 0 ]; then
    [ -s $ROOT/gates/C6-1_G.PASS ] || die "no $ROOT/gates/C6-1_G.PASS (C6-1's regression gate, c6_1_gate_v1.sh): refused"
  fi
  if [ -n "$RS_SLICE" ]; then   # in dry mode an absent resume input is reported, not refused (C6-1 or the source member makes it)
    local miss=""
    [ -s $RS_SLICE ] || miss="$miss $RS_SLICE"
    for f in .json _fields.npz _handoff.npz _spectra.npz _s11.npz; do [ -s $(dirname $RS_RUN)/collapse_$(basename $RS_RUN)$f ] || miss="$miss $(dirname $RS_RUN)/collapse_$(basename $RS_RUN)$f"; done
    if [ -n "$miss" ]; then [ "$MODE" = dry ] && echo "DRY: resume inputs absent (made by the source run):$miss" || die "resume inputs absent:$miss: refused"; fi
  fi; }

main_checks() {   # version 2: the device and the disk, main mode only
  if [ "$GPU" != cpu ]; then
    $PY -s -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || die "torch.cuda.is_available() is False under $PY (GPU=$GPU): refused"
  fi
  if [ "${DISK_ANY:-0}" != 1 ]; then
    mkdir -p $ROOT
    local avail need; avail=$(df -P -B1 $ROOT | awk 'NR==2{print $4}')
    need=$(awk -v n=$NSL -v s=$NSLICE_GB 'BEGIN{printf "%d", (n*s + 5)*1e9}')
    [ "$avail" -ge "$need" ] || die "free space on $ROOT is $(awk -v a=$avail 'BEGIN{printf "%.1f", a/1e9}') GB, the member needs $(awk -v b=$need 'BEGIN{printf "%.1f", b/1e9}') GB ($NSL slices at $NSLICE_GB GB plus 5): refused"
  fi; }

case $MODE in
  print) echo "$ENGINE ${ARGS[*]}";;
  dry)   # the engine's own argparse and main() checks with run() and C5.backend stubbed; then every kit module imported, each from the kit
    checks
    cd /tmp && env -u PYTHONPATH $PY -I -c "
import sys, os; kit = os.path.realpath(sys.argv[1]); sys.path.insert(0, kit); sys.argv = [sys.argv[2]] + sys.argv[3:]
import engine_collapse_v36b as E
E.run = lambda a: print('DRY OK: engine args parsed and checked; would run with', sorted((k, v) for k, v in vars(a).items() if v not in (None, False)))
E.C5.backend = lambda g: print('DRY: backend(gpu %s) stubbed' % g)
E.main()
import importlib, glob
for p in sorted(glob.glob(os.path.join(kit, 'engine_*.py'))): importlib.import_module(os.path.basename(p)[:-3])
bad = [(n, f) for n, m in list(sys.modules.items()) for f in [getattr(m, '__file__', None) or ''] if n.startswith('engine_') or n.startswith('compare_')]
bad = [(n, f) for n, f in bad if os.path.dirname(os.path.realpath(f)) != kit]
loc = [(n, getattr(m, '__file__', '')) for n, m in list(sys.modules.items()) if getattr(m, '__file__', None) and not os.path.realpath(m.__file__).startswith((os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix), kit))]
print('DRY imports: %d engine modules loaded, all from the kit: %s; modules from outside the kit and the interpreter: %s' % (sum(1 for n in sys.modules if n.startswith('engine_')), not bad, loc or 'none'))
sys.exit(1 if (bad or loc) else 0)" $C $ENGINE "${ARGS[@]}";;
  main)
    checks
    main_checks
    for f in $OUT $LOG $PIDF $PIDF.rc; do test ! -e $f || die "refusing: $f exists"; done
    mkdir -p $OUT $STD $ROOT/logs $ROOT/pids
    echo "$(now) start $M: $ENGINE ${ARGS[*]}" | tee $OUT/launch_line.txt
    TH=${THREADS:-8}; export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH
    TO=""; [ -n "${DEADLINE:-}" ] && TO="timeout $DEADLINE"
    cd $OUT
    QA=$(printf '%q ' "${ARGS[@]}")
    setsid nohup bash -c "env -u PYTHONPATH $TO nice -n 10 $PY -s -B $ENGINE $QA > $LOG 2>&1 < /dev/null & echo \$! > $PIDF; wait \$!; echo \$? > $PIDF.rc" > /dev/null 2>&1 < /dev/null &
    sleep 3; echo "$(now) launched $M tag $TAG pid $(cat $PIDF) log $LOG";;
  *) die "usage: $0 <member> main|print|dry";;
esac
