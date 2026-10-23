#!/bin/bash
# b1_card_v3b.sh MODE [BASE]  (busi lane, B1 ladder extension; ready_jobs_v3 section 32; the solve lane queues it)
# v3b over v3 (cd387bde), reviewer A's ruling busi/notes/REVIEWA_B1_ladder_ext_v1.md: KIT is $BASE/kit_v3b (kit_v3's 25 .py files unchanged plus
# engine_b1_tests_v3b.py); the new mode ladext runs the ruling's line (21.2's engine call with --ladder 2.4,3.0,4.7407,9.4815 and --T-over-rs 32,
# tag b1ladx, out runs/ladder_ext) with the ruling's stop rule across members: after each member's LADDER line, engine_b1_tests_v3b.py ladext
# decides; at the first member whose R_t is non-null at 32 r_s the driver kills ITS OWN engine child (the pid it started, kill by pid; nothing
# else is ever killed), keeps the finished members, and writes runs/ladder_ext/collapse_b1ladx_ladder_ext.json (E_trap, E_hold, c_E(0.4),
# the branch, rs_B1 = 2 g x 1.2 E_hold in branch (i)). The member lines are appended to runs/ladder/ladder.log (its original kept once as
# ladder.log.pre_ext). ksea reads that decision instead of 21.2's ladder json; bench, ladder, nolight and coupled100 are v3's commands verbatim.
# DRY=1 (ladext only): the md5 check, the python versions, the options parsed by engine_b1_v3's own parser and each member's r_s, tau and
# records printed; no card, disk or memory check, nothing run.
# MODE: bench | ladder | ladext | ksea | nolight | coupled100
# BASE (default ${QI2_ROOT}/busi_b1): the kit is $BASE/kit_v3b (code/ flat, MD5SUMS), the outputs $BASE/runs/<mode>/ (ladext: $BASE/runs/ladder_ext/).
# Every mode: checks the kit's md5s (md5sum -c), python with torch CUDA, the free disk and the card's free memory; refuses (exit 1) if its
# output directory already holds a result; runs at nice 10 with 16 CPU threads; writes $BASE/runs/<mode>/DONE (rc 0) or FAILED (rc != 0).
# v3 over v2 (ac4d6934): the kit_v3 files (reviewer A's red team REVIEWA_B1v2_redteam_v1.md applied); RT-17 the host's MemAvailable checked (60 GB for
# bench, whose 33,792-site init parks the states in pageable host memory; 24 GB for coupled100's 16,896-site offload, the lane's choice).
set -u
MODE=${1:?mode}
BASE=${2:-${QI2_ROOT}/busi_b1}
KIT=$BASE/kit_v3b
C=$KIT/code
OUT=$BASE/runs/$MODE
[ "$MODE" = ladext ] && OUT=$BASE/runs/ladder_ext   # v3b: the ruling's output directory
PY=${PYBIN:-python3}
export B1_PATHS=$C OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMPY_MADVISE_HUGEPAGE=0
die() { echo "b1_card_v3b $MODE: REFUSED: $*"; exit 1; }
DRY=${DRY:-0}
(cd $KIT && md5sum -c --quiet MD5SUMS) || die "kit md5 check failed ($KIT/MD5SUMS)"
if [ "$DRY" = 1 ]; then
  [ "$MODE" = ladext ] || die "DRY=1 is for ladext only"
  $PY -c "import numpy,scipy,torch,sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
else
$PY -c "import numpy,scipy,torch,sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__); sys.exit(0 if torch.cuda.is_available() else 1)" || die "no CUDA torch in $PY"
avail_gb=$(df -BG --output=avail $BASE | tail -1 | tr -dc 0-9)
gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
need_gpu=20000; need_disk=20; need_ram=0
case $MODE in bench|coupled100) need_gpu=76000; need_disk=40;; esac
case $MODE in bench) need_ram=60;; coupled100) need_ram=24;; esac                     # v3 RT-17
ram_gb=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)                  # v3 RT-17
[ "$ram_gb" -ge "$need_ram" ] || die "host MemAvailable $ram_gb GB, need $need_ram (the M6 offload parks the states in host memory)"   # v3 RT-17
[ "$avail_gb" -ge "$need_disk" ] || die "$avail_gb GB free on $BASE, need $need_disk"
[ "$gpu_free" -ge "$need_gpu" ] || die "card has $gpu_free MiB free, need $need_gpu (the mode wants the card alone)"
[ -e $OUT/DONE ] && die "$OUT/DONE exists"
fi
# the B1 point's common options (design 2.1; the ladder fixes --rs, ksea fixes --nsites and --kappa-sea)
COMMON="--m 0.4 --g 0.25 --R0-over-rs 4 --ell 2 --reading A --reference R2 --site-fields averaged --mix plain --maxit 20 --stop-E 1e-7 --stop-fields 1e-6 --delta-rtol-every 10 --commutator-every 10 --gpu 0"
# v3b: the ladder extension's engine arguments (the ruling's line: 21.2's with --ladder and --T-over-rs substituted; tag b1ladx)
LADX_F=2.4,3.0,4.7407,9.4815
LADX_ARGS="$COMMON --ladder $LADX_F --ladder-cE 0.135 --ladder-tau-over-rs 0.125 --nsites 1024 --kappa-sea 10 --T-over-rs 32 --rs 8.4375 --tau 1.0546875 --light off --io chunked --b1-save-every-rs 0 --tag b1ladx"
if [ "$DRY" = 1 ]; then
  echo "DRY ladext: OUT $OUT; engine: nice -n 10 $PY $C/engine_b1_v3.py $LADX_ARGS --out $OUT"
  $PY - $LADX_ARGS --out $OUT <<'PYEOF' || die "the options do not parse"
import sys, os
sys.path.insert(0, os.environ["B1_PATHS"])
import engine_b1_v3 as B
a = B.build_parser().parse_args(sys.argv[1:])
mP = a.g ** -0.5; Emax = a.ladder_cE * mP ** 3 / a.m ** 2
print("DRY parsed: m %g g %g nsites %d kappa_sea %d T_over_rs %g ladder_tau_over_rs %g light %s io %s tag %s; E_max %.6g" % (a.m, a.g, a.nsites, a.kappa_sea, a.T_over_rs, a.ladder_tau_over_rs, a.light, a.io, a.tag, Emax))
for fs in a.ladder.split(","):
    rs = 2.0 * a.g * float(fs) * Emax; tau = a.ladder_tau_over_rs * rs; n = int(round(a.T_over_rs * rs / tau))
    print("DRY member f %s: E0_rest %.6g r_s %.6g a tau %.6g a steps %d records %d (to T/r_s %.6g) tag %s_f%s" % (fs, float(fs) * Emax, rs, tau, n, n + 1, n * tau / rs, a.tag, fs))
PYEOF
  echo "DRY stop rule: after each LADDER line, $C/engine_b1_tests_v3b.py ladext --out $OUT --tag b1ladx --members $LADX_F --T-over-rs 32 --m 0.4 --g 0.25; first member held at 32 r_s -> kill the engine pid; else the next member"
  exit 0
fi
mkdir -p $OUT
ENG="nice -n 10 $PY $C/engine_b1_v3.py"
TST="nice -n 10 $PY $C/engine_b1_tests_v3.py"
TSTB="nice -n 10 $PY $C/engine_b1_tests_v3b.py"   # v3b: ladext only
# (COMMON, the B1 point's common options, design 2.1, is set above in v3b, unchanged from v3)
params() {  # rs nsites kappa_sea from ksea's B1_PARAMS.json
  P=$BASE/runs/ksea/B1_PARAMS.json; [ -s $P ] || die "no $P (run ksea first)"
  read RS NS KS < <($PY -c "import json;d=json.load(open('$P'));print(d['rs'],d['nsites'],d['kappa_sea'])")
  [ -n "$RS" ] && [ "$RS" != None ] || die "B1_PARAMS.json has no rs"
  TAU=$($PY -c "print(0.125*$RS)")
}
rc=0
cd $OUT
case $MODE in
  bench)   # review M1: propagate_lean at 16,896 and 33,792 sites, tau 1.05 and 4.2; the T_0 vacuum by device eigh at both lengths
    nice -n 10 $PY $C/bench_b1_m1_v3.py --n 16896,33792 --taus 1.05,4.2 --init-n 16896,33792 --device cuda:0 --chunk 2048 --tol 1e-14 --threads 16 --out $OUT/b1_m1_bench.json > $OUT/bench.log 2>&1 || rc=$?
    ;;
  ladder)  # design 2.2.1 with the light off: the trapping threshold at ma 0.4 (a build input only, owner #306); 20 r_s per member, tau 0.125 r_s
    $ENG $COMMON --ladder 0.5,0.7,0.85,1.0,1.15,1.3,1.6,2.0 --ladder-cE 0.135 --ladder-tau-over-rs 0.125 --nsites 1024 --kappa-sea 10 --T-over-rs 20 --rs 8.4375 --tau 1.0546875 --light off --io chunked --b1-save-every-rs 0 --out $OUT --tag b1lad > $OUT/ladder.log 2>&1 || rc=$?
    [ $rc = 0 ] && $PY -c "import json;d=json.load(open('$OUT/collapse_b1lad_ladder.json'));print('LADDER', d.get('E_trap_rest'), d.get('c_E_ratio'), d.get('rs_B1'), d['decision'])" | tee -a $OUT/ladder.log
    ;;
  ladext)  # v3b, reviewer A REVIEWA_B1_ladder_ext_v1: the extension at ma 0.4, f 2.4, 3.0, 4.7407, 9.4815 (r_s 8.1, 10.125, 16, 32 a), each to 32 r_s at 0.125 r_s
    LOG0=$BASE/runs/ladder/ladder.log
    [ -s $LOG0 ] && [ ! -e $LOG0.pre_ext ] && cp -p $LOG0 $LOG0.pre_ext
    echo "# B1_ladder_ext (b1_card_v3b ladext, kit_v3b, $(date -u)): members $LADX_F to 32 r_s, out $OUT" >> $LOG0
    $ENG $LADX_ARGS --out $OUT > $OUT/ladder_ext.log 2>&1 &
    EPID=$!
    echo "$EPID" > $OUT/engine.pid
    echo "b1_card_v3b ladext: engine pid $EPID" | tee -a $OUT/driver.log
    seen=0; stopped=0
    while :; do
      alive=1; kill -0 $EPID 2>/dev/null || alive=0
      n=$(grep -c '^LADDER f ' $OUT/ladder_ext.log 2>/dev/null); n=${n:-0}
      if [ "$n" -gt "$seen" ]; then
        grep '^LADDER f ' $OUT/ladder_ext.log | sed -n "$((seen+1)),${n}p" >> $LOG0
        seen=$n
        $TSTB ladext --out $OUT --md $OUT/B1_ladder_ext.md --tag b1ladx --members $LADX_F --T-over-rs 32 --m 0.4 --g 0.25 > $OUT/ladext_decide.log 2>&1
        br=$($PY -c "import json;print(json.load(open('$OUT/collapse_b1ladx_ladder_ext.json'))['branch'])" 2>/dev/null)
        # the decision tool failed: the stop rule falls back to the engine's own line (R_t_last of the member's last record, at 32 r_s)
        [ -z "$br" ] && br=$($PY -c "import json;l=[x for x in open('$OUT/ladder_ext.log') if x.startswith('LADDER f ')][-1];r=json.loads(l.split(': ',1)[1]);print('held(engine line)' if r.get('R_t_last') is not None else 'not held(engine line)')" 2>/dev/null)
        echo "$(date -u +%H:%M:%S) members done $n, branch $br" | tee -a $OUT/driver.log
        case "$br" in "(i)"|"(ii)"|"held(engine line)")
          if [ $alive = 1 ] && [ "$n" -lt 4 ]; then kill $EPID 2>/dev/null; echo "stop rule: member $n held R_t at 32 r_s; killed engine pid $EPID (the rest not run)" | tee -a $OUT/driver.log; stopped=1; fi
          break;;
        esac
      fi
      [ $alive = 0 ] && break
      sleep 30
    done
    wait $EPID; erc=$?
    [ $stopped = 1 ] || rc=$erc
    $TSTB ladext --out $OUT --md $OUT/B1_ladder_ext.md --tag b1ladx --members $LADX_F --T-over-rs 32 --m 0.4 --g 0.25 > $OUT/ladext_decide.log 2>&1 || rc=$?
    grep '^LADEXT decision' $OUT/ladext_decide.log | tee -a $LOG0
    ;;
  ksea)    # design 2.2.3 (kappa_sea 6, 8 against 10) and review M3 b (512 and 256 sites against 1024) at the B1 point, light off, 20 r_s
    L=$BASE/runs/ladder_ext/collapse_b1ladx_ladder_ext.json; [ -s $L ] || die "no $L (run ladext first)"   # v3b: the extension's decision
    RS=$($PY -c "import json,sys;d=json.load(open('$L'));sys.exit(1) if not str(d.get('decision','')).startswith('B1 point') else print(d['rs_B1'])") || die "the ladder's decision is not the B1 point: $($PY -c "import json;print(json.load(open('$L'))['decision'])")"
    TAU=$($PY -c "print(0.125*$RS)")
    for spec in ref:1024:10 ks8:1024:8 ks6:1024:6 ns512:512:10 ns256:256:10; do
      t=${spec%%:*}; r=${spec#*:}; n=${r%%:*}; k=${r#*:}
      $ENG $COMMON --rs $RS --tau $TAU --nsites $n --kappa-sea $k --T-over-rs 20 --light off --io chunked --b1-save-every-rs 0 --out $OUT --tag b1p_$t > $OUT/b1p_$t.log 2>&1 || { rc=$?; break; }
    done
    [ $rc = 0 ] && { $TST ksea --out $OUT --md $OUT/B1_ksea.md --ref $OUT/b1p_ref --ks $OUT/b1p_ks8,$OUT/b1p_ks6 --ns $OUT/b1p_ns512,$OUT/b1p_ns256 --rs-b1 $RS > $OUT/ksea_rule.log 2>&1 || rc=$?; cat $OUT/B1_PARAMS.json; }
    ;;
  nolight) # design 4.5 = D2 criterion 1: light off, 800 records (100 r_s) at the chosen nsites and kappa_sea
    params
    $ENG $COMMON --rs $RS --tau $TAU --nsites $NS --kappa-sea $KS --T-over-rs 100 --light off --io chunked --b1-save-every-rs 50 --json-every 80 --out $OUT --tag b1nl > $OUT/nolight.log 2>&1 || rc=$?
    [ $rc = 0 ] && $TST d2 --out $OUT --md $OUT/B1_D2_nolight.md --nolight $OUT/b1nl --rs0 $RS > /dev/null 2>&1; tail -4 $OUT/B1_D2_nolight.md 2>/dev/null
    ;;
  coupled100) # design 4.15 + review M2/M3: the coupled run's first 100 records on the 16,896-site light chain (kappa 1..3 sourced, 4 guard, 0 the R8 copy)
    params
    $ENG $COMMON --rs $RS --tau $TAU --nsites $NS --kappa-sea $KS --T-over-rs 100 --light on --light-nsites 16896 --light-kappas 1,2,3 --light-guard-kappas 4 --light-kappa-zero on \
      --light-init device --light-init-cache $OUT/init_cache --light-threads 16 --light-cheb-tol 1e-14 --light-chunk 2048 --light-exterior vacuum --light-ref continued \
      --tau-schedule on --tau-switch-rs 50 --tau-late 4.2 --benchmark-records 100 --b1-save-every-rs 0 --json-every 20 --fields-chunk-records 100 \
      --out $OUT --tag b1c100 > $OUT/coupled100.log 2>&1 || rc=$?
    $TST d2 --out $OUT --md $OUT/B1_D2_coupled100.md --coupled $OUT/b1c100 --rs0 $RS > /dev/null 2>&1; tail -12 $OUT/B1_D2_coupled100.md 2>/dev/null
    ;;
  *) die "unknown mode";;
esac
if [ $rc = 0 ]; then date -u > $OUT/DONE; else echo "rc $rc $(date -u)" > $OUT/FAILED; fi
(cd $OUT && find . -maxdepth 1 -type f ! -name '*.pt' -newer $KIT/MD5SUMS -print0 | xargs -0 md5sum > $OUT/MD5_outputs.txt)
echo "b1_card_v3b $MODE: rc $rc"
exit $rc
