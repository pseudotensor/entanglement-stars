#!/bin/bash
# restart_w256_rd1d_qh1_v1.sh (busi, inbox #464; ON qi2_h100_1): W256_s4 resumed from its T0001.50 slice (md5 736544ba) with run_c's json/fields/
# handoff/spectra (run_c ended by CUDA OOM at 16:50:12 UTC after T 2.0625; its json was last written at T 1.875) on RD-1d: engine
# engine_collapse_v36nq_rd13mfd.py 59a641d2 + engine_banded_eigh_v2.py 7010166e (code16d_rd1d = code16d's 84 files + those, + rd13mfb),
# --nk-stage-release on --eigh-module v2, cache off (--nk-cache-host off, as run_c), every other argument run_c's (args.txt) unchanged
# but the out, slice dir and resume pair. Out: W256_s4/run_d, states_d (run_c and states_c untouched).
# GATE (gate_w256_rd1d_v1.py beside it): the first json write after 1.5 must equal run_c's records at the same clocks BITWISE (json
# records by cmp_rd13m_v1's rule and the fields rows; RD-1d is bitwise against the parent's path), and nvidia-smi memory.used must stay
# <= 75 GiB throughout. On a FAIL the gate stops the engine by PID and this script starts the CACHE-ON FALLBACK: the parent engine
# rd13mf 248f25f3 (code16d) with --nk-cache-host on (about 1.7x slower), from T0001.50 with run_c, into run_e/states_e. Stopper: stop_flat_h100f_v1.py on the running run dir (caps T 12.5 and WALL_CAP).
set -u
W=${QI2_ROOT}/busi_gatecomp; M=$W/W256_s4; PY=${QI2_ROOT}/python; K=$W/w256rd1d
CODE=${QI2_ROOT}/code16d_rd1d; ENG=engine_collapse_v36nq_rd13mfd.py
CODE0=${QI2_ROOT}/code16d; ENG0=engine_collapse_v36nq_rd13mf.py
TAG=lad_W256_s4_m0p2_rs256_n4096_R04_k14_tau16_ell4_prep4_A_R2_avg_newton_v36nq_rd13mf
WALL_CAP=${WALL_CAP:?set WALL_CAP=YYYY-MM-DDTHH:MM:SSZ}
SL=$M/states_c/orbitals_${TAG}_T0001.50.npz; RC=$M/run_c
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $M/RESTART_W256_RD1D.txt; }
ck() { [ "$(md5sum < $1 | cut -c1-32)" = "$2" ] || { log "md5 $1 (want $2)"; exit 9; }; }
ck $CODE/$ENG 59a641d22773000956ec9abb1a50d4ca; ck $CODE/engine_banded_eigh_v2.py 7010166e889a86175c5813d1a2c2d96e
ck $CODE0/$ENG0 248f25f35e48a88a84b102ddd99a5713
ck $SL "$(awk '{print $1}' $K/SLICE_T0001.50.md5)"
(cd $K && md5sum -c --quiet MD5SUMS) || { log "kit MD5SUMS"; exit 9; }
for d in run_d states_d run_e states_e; do [ -e $M/$d ] && { log "$M/$d exists"; exit 9; }; done
mkdir -p $M/run_d $M/states_d $M/logs
BASE=$(sed -E -e 's# --resume-slice [^ ]+##' -e 's# --resume-run [^ ]+##' $RC/args.txt)
mkargs() {  # $1 run dir, $2 slice dir, $3 resume slice, $4 resume run, $5 extra
  echo "$BASE" | sed -E -e "s#--slice-dir [^ ]+#--slice-dir $2#" -e "s#--out [^ ]+#--out $1#" | sed "s#\$# $5 --resume-slice $3 --resume-run $4#"; }
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 NUMEXPR_NUM_THREADS=14 NUMBA_NUM_THREADS=14 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE; export PYTHONPATH=${QI2_ROOT}/peakhook
exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9; log "lock held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do log "GPU busy, waiting"; sleep 30; done
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 5000 >> $M/logs/smi_d.csv 2>&1 & SMI=$!
run_member() {  # $1 run dir, $2 slice dir, $3 code, $4 engine, $5 args; starts engine + watcher + stopper + pruner, waits
  local R=$1 S=$2; echo "$5" > $R/args.txt; md5sum $3/$4 > $R/RUN_MD5.txt
  ( cd $3 && exec $PY -B -u $3/$4 $5 ) > $R/log.txt 2>&1 < /dev/null &
  P=$!; echo $P > $R/pid; log "engine $4 pid $P into $R"
  $PY $W/flat/code/flat_stall_watch_v1.py $R/log.txt $P $M/logs/stallwatch_$(basename $R).tsv --from-T 1.5 > $M/logs/stallwatch_$(basename $R).out 2>&1 & WP=$!
  ( cd $M && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY $W/code/stop_flat_h100f_v1.py $M $TAG $3 --sigma 4 --period 300 \
      --t-cap 12.5 --wall-cap $WALL_CAP --run $(basename $R) --states $(basename $S) > $R/stop_flat.out 2>&1 ) & SP=$!
  log "stall watcher $WP, stopper $SP"
  ( while kill -0 $P 2>/dev/null; do   # rolling pruner as run_c's: the newest two, every 2 r_s, the dense 8.0 to 10.5 kept
      ls $S/orbitals_${TAG}_T*.npz 2>/dev/null | sed -E 's/.*_T([0-9.]+)\.npz$/\1/' | sort -g > $M/logs/slices_$(basename $S).txt
      n=$(wc -l < $M/logs/slices_$(basename $S).txt)
      head -n $(( n > 2 ? n - 2 : 0 )) $M/logs/slices_$(basename $S).txt | while read t; do
        awk -v t=$t 'BEGIN{x=t+0; keep=(x>=8.0-1e-9 && x<=10.5+1e-9) || (int(x/2+1e-9)*2==x)} END{exit !keep}' < /dev/null && continue
        f=$S/orbitals_${TAG}_T$t.npz; echo "$(date -u +%FT%TZ) pruned T$t $(md5sum < $f | cut -c1-32)" >> $M/logs/pruned_$(basename $S).txt; rm -f $f
      done; sleep 600; done ) &
}
A1=$(mkargs $M/run_d $M/states_d $SL $RC/$TAG "--nk-stage-release on --eigh-module v2")
run_member $M/run_d $M/states_d $CODE $ENG "$A1"
$PY $K/gate_w256_rd1d_v1.py $M $M/run_d $RC $TAG $P --t0 1.5 --period 60 --mem-cap-mib 76800 > $M/logs/gate_rd1d.out 2>&1 & log "gate pid $!"
wait $P; RC1=$?; log "run_d rc $RC1"; kill $WP $SP 2>/dev/null
if [ -e $M/GATE_FAIL.txt ]; then
  log "GATE FAIL ($(cat $M/GATE_FAIL.txt)): the cache-on fallback"
  RS="$SL $RC/$TAG"   # always from T0001.50 with run_c: the parent engine resumes its own run json (a gate fail comes at the first json write or the first heavy record)
  mkdir -p $M/run_e $M/states_e; [ -d $M/run_d ] && cp -a $M/run_d $M/run_d_gatefail
  A2=$(mkargs $M/run_e $M/states_e ${RS% *} ${RS#* } "" | sed 's#--nk-cache-host off#--nk-cache-host on#')
  log "fallback resume pair: $RS"
  run_member $M/run_e $M/states_e $CODE0 $ENG0 "$A2"
  wait $P; log "run_e rc $?"; kill $WP $SP 2>/dev/null
fi
kill $SMI 2>/dev/null; flock -u 9; log "lock released"
