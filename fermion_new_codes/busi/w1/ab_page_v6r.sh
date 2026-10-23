#!/bin/bash
# ab_page_v6r.sh (busi PERF PG-A1..A3, owner order #299): one arm of the Page A/B v6q against v6r on ONE card, the same arguments: the B2
# production line of launch_v6q_B2_card_v1.sh at --kappas K-K, a 2 r_s segment from T = 0 with the reference at 1 r_s, probes at 1, 1.5, 2
# and the columns at 1 and 2 r_s (so that the probes and the saved columns, PG-A3's consumers, run inside the segment). nice 10, 4 threads
# (the launcher's), the bench_protocol 4.1 item 5 telemetry (smi.csv, dmon.log, cpu.log, pid.txt), stopped by their own PIDs. Never kills an engine.
# MODE: A = page_history_v6q.py; B = page_history_v6r.py at its defaults (PG-A1 and PG-A3 on, the propagation chunk = --chunk 2048);
#       B256, B512, B128 = v6r with --prop-chunk 256/512/128 (PG-A2 sweep); Boff = v6r with --v6r-share-traces off --v6r-full-cache off.
# Usage: bash ab_page_v6r.sh MODE [GPU 0] [KAPPA 64]   (PY= to override the venv; PAGE= the page dir, default ${QI2_ROOT}/page)
MODE=$1; GPU=${2:-0}; KA=${3:-64}
PAGE=${PAGE:-${QI2_ROOT}/page}; PY=${PY:-${QI2_ROOT}/python}
cd $PAGE || exit 2
case $MODE in
  A) ENG=page_history_v6q.py; WANT=b4f4ea69c199ac9785e676e060f36a15; EXTRA="";;
  B) ENG=page_history_v6r.py; WANT=6935a5b296cdd9f2f3ac8f8108c0b050; EXTRA="";;
  B256|B512|B128) ENG=page_history_v6r.py; WANT=6935a5b296cdd9f2f3ac8f8108c0b050; EXTRA="--prop-chunk ${MODE#B}";;
  Boff) ENG=page_history_v6r.py; WANT=6935a5b296cdd9f2f3ac8f8108c0b050; EXTRA="--v6r-share-traces off --v6r-full-cache off";;
  *) echo "MODE must be A, B, B256, B512, B128 or Boff"; exit 2;;
esac
GOT=$(md5sum code_d13/$ENG | cut -d" " -f1)
[ "$GOT" = "$WANT" ] || { echo "REFUSED: code_d13/$ENG md5 $GOT, expected $WANT"; exit 3; }
OUT=bench/PGA_k${KA}_$MODE
[ -e $OUT ] && { echo "REFUSED: $OUT exists (fresh output directories only)"; exit 4; }
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=$GPU NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=$PAGE/code_v6:$PAGE/code_d13 PAGE_CODE_V6=$PAGE/code_v6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${QI2_ROOT}/B2; T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $OUT/CARD.txt
nvidia-smi -i $GPU --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm --format=csv -l 1 > $OUT/smi.csv 2>&1 & P1=$!
nvidia-smi dmon -i $GPU -s pucm -d 1 -o DT > $OUT/dmon.log 2>&1 & P2=$!
(command -v mpstat >/dev/null && mpstat -P ALL 5 || vmstat 5) > $OUT/cpu.log 2>&1 & P3=$!
echo "PGA_k${KA}_$MODE start $(date -u +%FT%TZ) $ENG $GOT gpu $GPU extra '$EXTRA'" | tee $OUT/ab.txt
nice -n 10 $PY code_d13/$ENG --record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --nsites 8192 --kappas $KA-$KA --kappa-hard-max $KA --no-kappa-extend --interp linear --interp-sub 8 --cheb-tol 1e-16 --sector auto --sector-propagate on --sector-check-every-rs 10 --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --collect-every-rs 2 --extra-ck-rs 15,30 --ext-every-rs 10 --vac-every-rs 10 --T-end-rs 2 --save-every 4 --device cuda:0 --threads 4 --out $OUT/run --tag PGA --resume --max-wall-hours 3 --columns-at-rs 2 --columns-every-rs 0 --ref-clocks-rs 1 --probe-from-rs 1 --probe-every-rs 0.5 $EXTRA > $OUT/log.txt 2>&1 &
EP=$!; echo $EP > $OUT/pid.txt; wait $EP; rc=$?
kill $P1 $P2 $P3 2>/dev/null
echo "PGA_k${KA}_$MODE exit $rc $(date -u +%FT%TZ)" | tee -a $OUT/ab.txt
exit $rc
