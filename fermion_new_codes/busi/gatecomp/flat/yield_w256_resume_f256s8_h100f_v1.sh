#!/bin/bash
# yield_w256_resume_f256s8_h100f_v1.sh (busi, GATECOMP; ReviewA_rz256ext_v1 sec. 1, inbox #427). ON h100-f.
#  1. W256_s4 (GC_W256_s4c, run_c/states_c) YIELDS at its NEXT slice: the first slice newer than the newest now whose clock the run json holds
#     (the json is rewritten every 5 records) and whose size is stable for 60 s; its engine (W_EP), stall watcher (W_WP) and stopper (W_SP)
#     are stopped by PID (TERM, KILL after 120 s; never pkill); run_c is copied to run_c_yieldT<t>/ (md5); the resume point is written.
#  2. The shared lock, then F256_s8 RESUMES from its T0004.50 slice: the same engine 02a1729a and line (lines_flat_taperW05/F256_s8.args
#     5951bb3d) with --T-over-rs 13.0, --source-continuation two-stage, slices every 0.5 r_s from 5.0 to 13.0 (ROLLING: the newest two and
#     every 2 r_s kept past 4.5; the slices to 4.5 are kept for the controls), into the same run/ from a copy run_preT0004.50/; labelled
#     post-gate, class C, "conditional on the interior's completion, root-gated"; with the stall watcher and stop_flat_h100f_v2.py --mode
#     postgate (the root gate on [R_t, 2 R_t] and the root's closing as stops; the charge alert at 3 percent; caps T 13.5 and 36 h).
set -u
W=${QI2_ROOT}/busi_gatecomp; MW=$W/W256_s4; M=$W/flat/F256_s8; PY=${QI2_ROOT}/python
W_EP=${W_EP:?}; W_WP=${W_WP:-}; W_SP=${W_SP:-}
TW=lad_W256_s4_m0p2_rs256_n4096_R04_k14_tau16_ell4_prep4_A_R2_avg_newton_v36nq_rd13mf
T8=lad_F256_s8_fbt05_m0p2_rs256_n4096_R04_k15_tau16_ell8_prep4_A_R2_avg_newton_v36nq_rd13mfb
CODE=${QI2_ROOT}/code16d_flat; ENG=engine_collapse_v36nq_rd13mfb.py; A=$W/flat/lines_flat_taperW05/F256_s8.args
[ "$(md5sum < $A | cut -c1-32)" = 5951bb3d64c5a86ee6e0915081a41462 ] || { echo "args md5"; exit 9; }
[ "$(md5sum < $CODE/$ENG | cut -c1-32)" = 02a1729ab808948d443eed9144b75b03 ] || { echo "engine md5"; exit 9; }
SL8=$M/states/orbitals_${T8}_T0004.50.npz; [ -e $SL8 ] || { echo "no $SL8"; exit 9; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $M/RESUME_POST.txt; }
ylog() { echo "$(date -u +%FT%TZ) $*" | tee -a $MW/YIELD_c.txt; }
newest() { ls $MW/states_c/orbitals_${TW}_T*.npz 2>/dev/null | sed -E 's/.*_T([0-9.]+)\.npz$/\1/' | sort -g | tail -1; }
N0=$(newest); ylog "yield requested (ReviewA_rz256ext_v1); engine pid $W_EP; newest slice now T$N0"
until N=$(newest) && [ -n "$N" ] && [ "$N" != "$N0" ] && f=$MW/states_c/orbitals_${TW}_T$N.npz && s1=$(stat -c %s $f) && sleep 60 && [ "$s1" = "$(stat -c %s $f)" ] && \
      $PY -c "import json,sys; d=json.load(open('$MW/run_c/collapse_$TW.json')); sys.exit(0 if max(r['T_over_rs'] for r in d['records']) >= float('$N') - 1e-9 else 1)" 2>/dev/null; do sleep 30; done
ylog "slice T$N complete (md5 $(md5sum < $f | cut -c1-32)), the json holds it; stopping engine $W_EP, watcher $W_WP, stopper $W_SP"
desc() { for c in $(ps -o pid= --ppid $1); do echo $c; desc $c; done; }
KIDS=$(desc $W_EP | tr '\n' ' '); kill -TERM $W_EP; for i in $(seq 1 24); do kill -0 $W_EP 2>/dev/null || break; sleep 5; done; kill -0 $W_EP 2>/dev/null && { kill -KILL $W_EP; ylog "KILL sent"; }
for k in $KIDS $W_WP $W_SP; do [ -n "$k" ] && kill -0 $k 2>/dev/null && kill -TERM $k; done
Y=$MW/run_c_yieldT$N; mkdir -p $Y; cp -p $MW/run_c/collapse_${TW}* $MW/run_c/log.txt $MW/run_c/args.txt $Y/; md5sum $Y/* > $Y/MD5SUMS
ylog "YIELDED at T$N; last record $(grep -E '^T/rs [0-9.]+ E/E0' $MW/run_c/log.txt | tail -1 | cut -c1-20); RESUME: the args of run_c/args.txt with --resume-slice $f --resume-run $Y/$TW (into run_c), after F256_s8"
sleep 10
exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9; log "lock held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "GPU busy, waiting"; sleep 15; done
P0=$M/run_preT0004.50; mkdir -p $P0; cp -p $M/run/collapse_${T8}* $M/run/log.txt $P0/; md5sum $P0/* > $P0/MD5SUMS
SLN=$(for x in $(seq 5.0 0.5 13.0); do printf "%.4f," $x; done | sed 's/,$//')
ARGS1="$(sed -e "s#--slices-c128 [^ ]*#--slices-c128 $SLN#" -e "s#--T-over-rs 6.5#--T-over-rs 13.0#" $A) --source-continuation two-stage --resume-slice $SL8 --resume-run $P0/$T8"
echo "$ARGS1" > $M/run/args_post_T4.50.txt; md5sum $CODE/$ENG $SL8 $P0/collapse_$T8.json > $M/run/RUN_MD5_post_T4.50.txt
echo "post-gate, class C: conditional on the interior's completion, root-gated (ReviewA_rz256ext_v1); resumed from T0004.50 to T 13.0 (cap 13.5 or 36 card-h)" > $M/LABEL.txt
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 NUMEXPR_NUM_THREADS=14 NUMBA_NUM_THREADS=14 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE; export PYTHONPATH=${QI2_ROOT}/peakhook
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 5000 >> $M/logs/smi.csv 2>&1 & SMI=$!
mv $M/run/log.txt $M/run/log_preT4.50.txt
( cd $CODE && exec $PY -B -u $CODE/$ENG $ARGS1 ) > $M/run/log.txt 2>&1 < /dev/null &
P=$!; echo $P > $M/run/pid; log "F256_s8 resumed from T0004.50 (post-gate, class C), engine pid $P"
$PY $W/flat/code/flat_stall_watch_v1.py $M/run/log.txt $P $M/logs/stallwatch_post.tsv --from-T 4.5 > $M/logs/stallwatch_post.out 2>&1 & log "stall watcher pid $!"
( cd $M && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY $W/code/stop_flat_h100f_v2.py $M $T8 $CODE --sigma 8 --mode postgate --from-T 4.5 --charge-line 0.03 --t-cap 13.5 --wall-cap 2026-10-24T00:00:00Z --period 300 > $M/stop_post.out 2>&1 ) & log "postgate stopper pid $!"
( while kill -0 $P 2>/dev/null; do
    ls $M/states/orbitals_${T8}_T*.npz 2>/dev/null | sed -E 's/.*_T([0-9.]+)\.npz$/\1/' | sort -g | awk '$1>4.5+1e-9' > $M/logs/slices_post.txt
    n=$(wc -l < $M/logs/slices_post.txt)
    head -n $(( n > 2 ? n - 2 : 0 )) $M/logs/slices_post.txt | while read t; do
      awk -v t=$t 'BEGIN{x=t+0; exit !(int(x/2+1e-9)*2==x)}' < /dev/null && continue
      f=$M/states/orbitals_${T8}_T$t.npz; echo "$(date -u +%FT%TZ) pruned T$t $(md5sum < $f | cut -c1-32)" >> $M/logs/pruned_post.txt; rm -f $f
    done; sleep 600; done ) &
wait $P; log "F256_s8 post-gate run rc $?"; kill $SMI; flock -u 9; log "lock released"
