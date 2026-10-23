#!/bin/bash
# restart_f256s8_cache_v1.sh (busi f52b1152, inbox #443): F256_s8 died OOM at 07:56 UTC 22 Oct (continuation keeps an extra
# evaluation cache on the device; RD-1d finding). Resume from T0004.50 exactly as yield_w256_resume_f256s8_h100f_v1.sh did,
# plus --nk-cache-host on (the OOM fallback both rulings name). ON h100-f.
set -u
W=${QI2_ROOT}/busi_gatecomp; M=$W/flat/F256_s8; T8=lad_F256_s8_fbt05_m0p2_rs256_n4096_R04_k15_tau16_ell8_prep4_A_R2_avg_newton_v36nq_rd13mfb
PY=${QI2_ROOT}/python; CODE=${QI2_ROOT}/code16d_flat; ENG=engine_collapse_v36nq_rd13mfb.py
SL8=$M/states/orbitals_${T8}_T0004.50.npz; P0=$M/run_preT0004.50
log(){ echo "$(date -u +%FT%TZ) $*" >> $M/RESTART_CACHE.txt; }
C=$M/run_crash_0756; mkdir -p $C; cp -p $M/run/collapse_${T8}* $M/run/log.txt $M/run/args_post_T4.50.txt $C/ 2>/dev/null; md5sum $C/* > $C/MD5SUMS; log "crashed run saved to $C"
A=$M/run/args_post_T4.50.txt; ARGS2="$(cat $A) --nk-cache-host on"
echo "$ARGS2" > $M/run/args_post_T4.50_cache.txt; md5sum $CODE/$ENG $SL8 $P0/collapse_$T8.json > $M/run/RUN_MD5_post_T4.50_cache.txt
echo "post-gate, class C: conditional on the interior's completion, root-gated (ReviewA_rz256ext_v1); resumed from T0004.50 to T 13.0 (cap 13.5 or 36 card-h); RESTART after OOM 07:56 with --nk-cache-host on (#443)" > $M/LABEL.txt
exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9; log "lock held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "GPU busy, waiting"; sleep 15; done
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 NUMEXPR_NUM_THREADS=14 NUMBA_NUM_THREADS=14 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE; export PYTHONPATH=${QI2_ROOT}/peakhook
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 5000 >> $M/logs/smi.csv 2>&1 & SMI=$!
mv $M/run/log.txt $M/run/log_crash_0756.txt
( cd $CODE && exec $PY -B -u $CODE/$ENG $ARGS2 ) > $M/run/log.txt 2>&1 < /dev/null &
P=$!; echo $P > $M/run/pid; log "F256_s8 restarted from T0004.50 with --nk-cache-host on, engine pid $P"
$PY $W/flat/code/flat_stall_watch_v1.py $M/run/log.txt $P $M/logs/stallwatch_post2.tsv --from-T 4.5 > $M/logs/stallwatch_post2.out 2>&1 & log "stall watcher pid $!"
( cd $M && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY $W/code/stop_flat_h100f_v2.py $M $T8 $CODE --sigma 8 --mode postgate --from-T 4.5 --charge-line 0.03 --t-cap 13.5 --wall-cap 2026-10-24T00:00:00Z --period 300 > $M/stop_post2.out 2>&1 ) & log "postgate stopper pid $!"
wait $P; log "F256_s8 run rc $?"; kill $SMI; flock -u 9; log "lock released"
