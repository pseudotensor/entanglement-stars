#!/bin/bash
# dissolve_launch_v1.sh (busi f52b1152, 2026-10-20, inbox #323): the launcher of the DISSOLVE card members of
# deliveries/lead_gaps/REVIEWA_write_DISSOLVE_v1.md, section "MERGED SPECIFICATION (for busi)" (binding). Every member is the stored (32, 4)
# history pair_rs32_ell4a_v36b's engine line (engine_collapse_v36b 9794c1b3, pod 5's copy) with the flags of the spec's table changed.
# Model: ${QI2_ROOT}/w2p_launch_busi.f4c_anc.sh (refuse if any output exists, nohup setsid, pid and rc file) and the
# *_dry modes of busi/code/iso/ready_jobs_v3e.sh (the engine's own argparse and checks with run() and the GPU backend stubbed).
# Usage: dissolve_launch_v1.sh <member> main|print|dry
#   members: G (C6-1), P0 (C6-3), X05 X20 (C6-4), L2 (C6-5), L8 (C6-6), S1 (C6-7), W (C6-8), S05 (C6-9), M64 (C6-11),
#            P40_L2 P40_L8 P40_S1 P40_S05 (C6-10, the continuation of that member from its T/r_s = 6 slice to 40),
#            PREP0 (the base line to T/r_s = 0: the preparation's known answer, k0, E0 and records[0].E_dust_shells)
# Environment: KIT (default ${QI2_ROOT}/kit_v1), DROOT (default ${QI2_ROOT}/dissolve: runs/, states/, logs/, pids/, gates/),
#   PY (default ${QI2_ROOT}/python: torch 2.8.0 is required, the massless preparation fails on torch 2.4.1, solve STATUS),
#   GPU (default 0; cpu omits --gpu, the box CPU known answer PREP0 only). THREADS (default 8). DEADLINE (seconds; the engine under timeout;
#   unset on the cards, where the solve lane's queue governs). GATE=0 skips the C6-1 PASS-file requirement (dry checks only). TORCH_ANY=1 skips the torch 2.8.0 check (dry only).
# The engine runs with only the kit's code directory on its path: python -s -B with PYTHONPATH unset, cwd the member's own empty run dir.
# This script never kills anything and never touches a queue file.
set -u
KIT=${KIT:-${QI2_ROOT}/kit_v1}; ROOT=${DROOT:-${QI2_ROOT}/dissolve}; PY=${PY:-${QI2_ROOT}/python}; GPU=${GPU:-0}
C=$KIT/code; ENGINE=$C/engine_collapse_v36b.py; MENG=9794c1b38fe59d19f2bb510673e757f2
BT=pair_rs32_ell4a_v36b
M=${1:-}; MODE=${2:-}
now() { date -u +%FT%TZ; }
die() { echo "$(now) dissolve_launch_v1 $M: $*"; exit 1; }

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
SL6=1,2,3,4,5,6
member() {   # $1 member: sets TAG, SL and the changed flags (the spec's table)
  case $1 in
    G)   TAG=${BT}_G;   SL=1.75,1.8125,2.0,2.3125; setf --T-over-rs 2.625;;
    P0)  TAG=${BT}_P0;  SL=""; setf --reference R1; setf --T-over-rs 3.0;;
    X05) TAG=${BT}_X05; SL=""; setf --c-LZ 0.5; setf --T-over-rs 3.0;;
    X20) TAG=${BT}_X20; SL=""; setf --c-LZ 2.0; setf --T-over-rs 3.0;;
    L2)  TAG=${BT}_L2;  SL=$SL6; setf --prep-ell 4; setf --ell 2; setf --T-over-rs 6.0;;
    L8)  TAG=${BT}_L8;  SL=$SL6; setf --prep-ell 4; setf --ell 8; setf --T-over-rs 6.0;;
    S1)  TAG=${BT}_S1;  SL=$SL6; setf --tau 1.0; setf --T-over-rs 6.0;;
    S05) TAG=${BT}_S05; SL=$SL6; setf --tau 0.5; setf --T-over-rs 6.0;;
    W)   TAG=${BT}_W;   SL=""; setf --T-over-rs 2.625
         RS_SLICE=$ROOT/states/${BT}_G/orbitals_${BT}_G_T0002.00.npz; RS_RUN=$ROOT/runs/${BT}_G/${BT}_G
         setf --resume-slice $RS_SLICE; setf --resume-run $RS_RUN; setf --weyl-certified eigvals; setf --segment-end-T-over-rs 2.625;;
    P40_L2|P40_L8|P40_S1|P40_S05)
         local s=${1#P40_}; member $s; local ST=$TAG; TAG=${ST}_P40; SL=10,20,30,40; setf --T-over-rs 40.0
         RS_SLICE=$ROOT/states/$ST/orbitals_${ST}_T0006.00.npz; RS_RUN=$ROOT/runs/$ST/$ST
         setf --resume-slice $RS_SLICE; setf --resume-run $RS_RUN;;
    PREP0) TAG=${BT}_prep0_cpu; SL=""; setf --T-over-rs 0.0;;   # the preparation's known answer (box CPU, GPU=cpu): record 0 only
    M64) TAG=pair_rs64_ell8a_v36b_M64; SL=$SL6; setf --rs 64.0; setf --nsites 4096; setf --ell 8; setf --ball-orbitals 1:24
         setf --ball-E-target 128.0; setf --T-over-rs 6.0;;
    *) die "unknown member '$1' (G P0 X05 X20 L2 L8 S1 W S05 M64 P40_L2 P40_L8 P40_S1 P40_S05)";;
  esac; }
RS_SLICE=""; RS_RUN=""
member "$M"
OUT=$ROOT/runs/$TAG; STD=$ROOT/states/$TAG; LOG=$ROOT/logs/$TAG.log; PIDF=$ROOT/pids/DISSOLVE_$M.pid
ARGS+=(--slices-c128 "$SL" --slice-dir $STD); [ "$GPU" = cpu ] || ARGS+=(--gpu $GPU); ARGS+=(--out $OUT --tag $TAG)

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
