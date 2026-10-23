#!/bin/bash
# rd13m_selftest_v1.sh (busi, inbox #387): the self-test of engine_collapse_v36nq_rd13m (--nk-cache-host) on the box GPU, fresh from
# T = 0 to T/r_s 0.125 (3 records) at NSITES (default 1024; the lad family's arguments, run_probe_v1.sh's scaling, K 10, the reopen2
# Newton block, pair workers 0), the end slice written. Arms, each its own process, THR threads (2), run together:
#   S0  engine_collapse_v36nq_rd13.py  (the source)          $EIGH
#   S1  engine_collapse_v36nq_rd13m.py (option off)          $EIGH        -> bitwise S0 (the defaults)
#   S2  engine_collapse_v36nq_rd13m.py --nk-cache-host on    $EIGH        -> bitwise S0 (the option)
# then cmp_rd13m_v1 S1 vs S0 and S2 vs S0 (records, head, fields, spectra, handoff, slices).
#   [ARMS=S0,S2] [THR=n] rd13m_selftest_v1.sh NAME [NSITES] [EIGH options, e.g. "--eigh banded"]
set -u
NAME=$1; NS=${2:-1024}; EIGH=${3:-}
M=${QI2_ROOT}/rd13m; PY=${QI2_ROOT}/python
GPU=${GPU:-1}; [ "$GPU" = 0 ] && { echo "never cuda:0"; exit 2; }
W=$M/selftest/$NAME; mkdir -p $W
RS=$((NS / 32)); ELL=$((RS / 16)); [ $ELL -lt 1 ] && ELL=1
COMMON="--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --kappa-sea 10 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --gpu 0 \
--site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes \
--mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 \
--nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001 --pair-workers 0 \
--rs $RS --nsites $NS --ell $ELL --prep-ell 4 --tau $ELL --T-over-rs 0.125 --slices-c128 0.125"
cd $M/code
md5sum engine_collapse_v36nq_rd13.py engine_collapse_v36nq_rd13m.py engine_banded_eigh_v1.py engine_fields_v16.py > $W/md5_inputs.txt
ALL=("S0 engine_collapse_v36nq_rd13.py" "S1 engine_collapse_v36nq_rd13m.py" "S2 engine_collapse_v36nq_rd13m.py --nk-cache-host on")
SEL=${ARMS:-S0,S1,S2}   # ARMS=S0,S2 runs the source and the option only
for A in "${ALL[@]}"; do
  case ",$SEL," in *",${A%% *},"*) ;; *) continue;; esac
  set -- $A; nm=$1; eng=$2; shift 2
  mkdir -p $W/$nm/states
  cmd="$PY -u $M/code/$eng $COMMON $EIGH --out $W/$nm --tag st_$nm --slice-dir $W/$nm/states $*"
  echo "$cmd" > $W/$nm/cmd.txt
  env CUDA_VISIBLE_DEVICES=$GPU NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=${THR:-2} MKL_NUM_THREADS=${THR:-2} OPENBLAS_NUM_THREADS=${THR:-2} NUMBA_NUM_THREADS=${THR:-2} \
      nice -n 10 $cmd > $W/$nm/log.txt 2>&1 &
done
wait
for nm in S1 S2; do
  [ -d $W/$nm ] || continue
  $PY $M/build/cmp_rd13m_v1.py $W/S0 st_S0 $W/$nm st_$nm --slices-a $W/S0/states --slices-b $W/$nm/states --out $W/cmp_${nm}_vs_S0.json > $W/cmp_${nm}_vs_S0.txt 2>&1
  echo "$nm vs S0 rc $?" >> $W/verdict.txt
done
grep -h "rd13m:" $W/S2/log.txt | head -3 >> $W/verdict.txt
echo "[$(date -u +%FT%TZ)] done" >> $W/verdict.txt
