#!/bin/bash
# b1_card_v3.sh MODE [BASE]  (busi lane, B1 build; ready_jobs_v3 sections 21.x; the solve lane queues it, this script never kills anything)
# MODE: bench | ladder | ksea | nolight | coupled100
# BASE (default ${QI2_ROOT}/busi_b1): the kit is $BASE/kit_v3 (code/ flat, MD5SUMS), the outputs $BASE/runs/<mode>/.
# Every mode: checks the kit's md5s (md5sum -c), python with torch CUDA, the free disk and the card's free memory; refuses (exit 1) if its
# output directory already holds a result; runs at nice 10 with 16 CPU threads; writes $BASE/runs/<mode>/DONE (rc 0) or FAILED (rc != 0).
# v3 over v2 (ac4d6934): the kit_v3 files (reviewer A's red team REVIEWA_B1v2_redteam_v1.md applied); RT-17 the host's MemAvailable checked (60 GB for
# bench, whose 33,792-site init parks the states in pageable host memory; 24 GB for coupled100's 16,896-site offload, the lane's choice).
set -u
MODE=${1:?mode}
BASE=${2:-${QI2_ROOT}/busi_b1}
KIT=$BASE/kit_v3
C=$KIT/code
OUT=$BASE/runs/$MODE
PY=${PYBIN:-python3}
export B1_PATHS=$C OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMPY_MADVISE_HUGEPAGE=0
die() { echo "b1_card_v3 $MODE: REFUSED: $*"; exit 1; }
(cd $KIT && md5sum -c --quiet MD5SUMS) || die "kit md5 check failed ($KIT/MD5SUMS)"
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
mkdir -p $OUT
ENG="nice -n 10 $PY $C/engine_b1_v3.py"
TST="nice -n 10 $PY $C/engine_b1_tests_v3.py"
# the B1 point's common options (design 2.1; the ladder fixes --rs, ksea fixes --nsites and --kappa-sea)
COMMON="--m 0.4 --g 0.25 --R0-over-rs 4 --ell 2 --reading A --reference R2 --site-fields averaged --mix plain --maxit 20 --stop-E 1e-7 --stop-fields 1e-6 --delta-rtol-every 10 --commutator-every 10 --gpu 0"
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
  ksea)    # design 2.2.3 (kappa_sea 6, 8 against 10) and review M3 b (512 and 256 sites against 1024) at the B1 point, light off, 20 r_s
    L=$BASE/runs/ladder/collapse_b1lad_ladder.json; [ -s $L ] || die "no $L (run ladder first)"
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
echo "b1_card_v3 $MODE: rc $rc"
exit $rc
