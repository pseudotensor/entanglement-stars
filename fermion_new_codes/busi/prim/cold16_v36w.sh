#!/bin/bash
# cold16_v36w.sh (busi, PERF W2-P3; REVIEWA_perf_w2_v1 P3 A/B (iii), W10 test 3's form): N (default 16) cold one-record launches of the box self-test's
# 512-site frozen-chart fixture (selftest_v36w_box.sh's fx_run fz line, ending at T/rs 0.125) on engine_collapse_v36w.py with --streams 1 on one GPU,
# each a fresh process. PASS = every launch rc 0, no "Traceback", no "stopped:" and no "v36w perf:" line in its log (solve.md S-1's gate).
# Usage: bash cold16_v36w.sh GPU [N]. Env: KIT (default ${QI2_ROOT}/w2perf), PY (default ${QI2_ROOT}/python), THREADS (default 8).
# Refuses if the output exists or the engine md5 differs. Never kills anything.
set -u -o pipefail
GPU=${1:?GPU index}; N=${2:-16}
KIT=${KIT:-${QI2_ROOT}/w2perf}; PY=${PY:-${QI2_ROOT}/python}; THREADS=${THREADS:-8}
CODE=$KIT/code; OUT=$KIT/bench/W2perf_cold16
ENG=engine_collapse_v36w.py; EMD5=aa91d291aee7191b3a4ca3f4640e465b
FIX="--m 0.2 --g 0.25 --rs 16 --R0-over-rs 4 --reading A --nsites 512 --kappa-sea 4 --reference R2 --site-fields averaged --ell 4 --prep-ell 4 --tau 2 --mix plain --T-over-rs 0.125 --slices-c128 0.125"
test ! -e $OUT || { echo "refusing: $OUT exists"; exit 1; }
m=$(md5sum $CODE/$ENG | cut -d' ' -f1); [ "$m" = "$EMD5" ] || { echo "refusing: md5 of $CODE/$ENG is $m, not $EMD5"; exit 1; }
mkdir -p $OUT; cd $CODE
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
ok=0
for i in $(seq -w 1 $N); do
  o=$OUT/c$i; mkdir -p $o
  nice -n 10 $PY $CODE/$ENG $FIX --v36-fixes --field-kernel-cache --out $o --tag fz --slice-dir $o/states --step dg --dg-chart frozen --dg-mix plain --dg-sub-max 0 --streams 1 --gpu $GPU > $o/fz.log 2>&1 < /dev/null; rc=$?
  bad=$(grep -c -E "Traceback|stopped:|v36w perf:" $o/fz.log)
  if [ $rc -eq 0 ] && [ "$bad" = 0 ]; then ok=$((ok + 1)); v=ok; else v=FAIL; fi
  echo "$(date -u +%FT%TZ) cold launch $i: rc $rc, bad lines $bad -> $v" | tee -a $OUT/cold16.log
done
echo "cold16: $ok of $N launches clean -> $([ $ok -eq $N ] && echo PASS || echo FAIL)" | tee -a $OUT/cold16.log
[ $ok -eq $N ]
