#!/bin/bash
# ab_w2_v36w.sh (busi, PERF W2-P3: --streams 1 made safe; ab_w2_v36u.sh with the v36w modes; final4/PERF/bench_protocol.md 4.1 item 5 telemetry, 4.4 rule 1
# gate): one W2 segment from the stored T0012.50 slice (md5 4aa640af) to the record at 12.625, the member's arguments word for word (w2_cmd_from_args.py on the
# member's json, resume keys dropped) and the telemetry wrapper. Usage: bash ab_w2_v36w.sh MODE GPU
#   MODE A   engine_collapse_v36p_fixv4c.py (md5 b623d3b0), the production engine
#   MODE B   engine_collapse_v36u.py --dg-eigh-cache-gib 18 --dg-node-workers 3 (v36u's B, the reference for WB; reuse ab_w2_v36u.sh's run if done)
#   MODE W0  engine_collapse_v36w.py (md5 aa91d291aee7191b3a4ca3f4640e465b) at the defaults (optional: v36w off = fixv4c; compare with A)
#   MODE WS  engine_collapse_v36w.py --streams 1 (P3 alone; compare with A)
#   MODE WB  engine_collapse_v36w.py --dg-eigh-cache-gib 18 --dg-node-workers 3 --streams 1 (P3 on top of P1/P2; compare with B: reviewer A P3 A/B (ii))
# Env: KIT (default ${QI2_ROOT}/w2perf: code/, slice/, member.json), PY (default ${QI2_ROOT}/python), THREADS (default 8, the
# production launcher's). Refuses if the output exists or an md5 differs. Never kills an engine; only its own telemetry samplers, by their pids.
set -u -o pipefail
MODE=${1:?MODE A, B, W0, WS or WB}; GPU=${2:?GPU index}
KIT=${KIT:-${QI2_ROOT}/w2perf}; PY=${PY:-${QI2_ROOT}/python}; THREADS=${THREADS:-8}
CODE=$KIT/code; SL=$KIT/slice; TAG=W2perf; OUT=$KIT/bench/W2perf_$MODE
RT=lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f
SLICE=$SL/orbitals_${RT}_T0012.50.npz; RUNR=$SL/$RT
case $MODE in
  A)  ENG=engine_collapse_v36p_fixv4c.py; EMD5=b623d3b010fa846cffcf914b4fa4900a; EXTRA="";;
  B)  ENG=engine_collapse_v36u.py; EMD5=062d7dd4700b839470fe3c5b70143740; EXTRA="--dg-eigh-cache-gib 18 --dg-node-workers 3";;
  W0) ENG=engine_collapse_v36w.py; EMD5=aa91d291aee7191b3a4ca3f4640e465b; EXTRA="";;
  WS) ENG=engine_collapse_v36w.py; EMD5=aa91d291aee7191b3a4ca3f4640e465b; EXTRA="--streams 1";;
  WB) ENG=engine_collapse_v36w.py; EMD5=aa91d291aee7191b3a4ca3f4640e465b; EXTRA="--dg-eigh-cache-gib 18 --dg-node-workers 3 --streams 1";;
  *) echo "unknown MODE $MODE"; exit 1;;
esac
test ! -e $OUT || { echo "refusing: $OUT exists"; exit 1; }
chk() { local m; m=$(md5sum "$1" | cut -d' ' -f1); [ "$m" = "$2" ] || { echo "refusing: md5 of $1 is $m, not $2"; exit 1; }; }
chk $CODE/$ENG $EMD5
chk $CODE/engine_pairsolve_v2.py 00d66bac1a9b39ed1503f35d23e5faf1
chk $CODE/assign_primitive_v1d.py 093e87879198cce4114a6229c1ccb11a
chk $SLICE 4aa640af43e5a1966be0de24ee22f88a
test -s $SL/collapse_${RT}.json || { echo "refusing: the resume run's json $SL/collapse_${RT}.json is missing"; exit 1; }
test -s $KIT/member.json || { echo "refusing: $KIT/member.json (the member's run json) is missing"; exit 1; }
MARGS=$(cd $CODE && $PY $CODE/w2_cmd_from_args.py $KIT/member.json) || { echo "refusing: w2_cmd_from_args failed"; exit 1; }
mkdir -p $OUT
cd $CODE
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
CMD="$PY $CODE/$ENG $MARGS --resume-slice $SLICE --resume-run $RUNR --segment-end-T-over-rs 12.625 --slices-c128 '' --out $OUT --tag $TAG --slice-dir $OUT/states --gpu $GPU $EXTRA"
{ echo "MODE $MODE GPU $GPU $(date -u +%FT%TZ)"; echo "$CMD"; md5sum $CODE/$ENG $CODE/engine_pairsolve_v2.py $CODE/assign_primitive_v1d.py $SLICE; nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv; nproc; free -g; } > $OUT/READ_head.txt 2>&1
# bench_protocol 4.1 item 5: the telemetry, started before the segment
nvidia-smi -i $GPU --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm --format=csv -l 1 > $OUT/smi.csv 2>&1 & P_SMI=$!
nvidia-smi dmon -i $GPU -s pucm -d 1 -o DT > $OUT/dmon.log 2>&1 & P_DMON=$!
( mpstat -P ALL 5 2>/dev/null || vmstat 5 ) > $OUT/cpu.log 2>&1 & P_CPU=$!
eval nice -n 10 $CMD > $OUT/log.txt 2>&1 < /dev/null & P_ENG=$!
echo $P_ENG > $OUT/pid.txt
echo "$(date -u +%FT%TZ) W2perf_$MODE started pid $P_ENG on GPU $GPU ($OUT)"
P_SPY=""
if command -v py-spy > /dev/null 2>&1; then
  ( sleep 180; kill -0 $P_ENG 2>/dev/null && py-spy record --pid $P_ENG --native --nonblocking --rate 100 --duration 240 --format speedscope -o $OUT/pyspy.json > $OUT/pyspy.log 2>&1 ) & P_SPY=$!
fi
wait $P_ENG; RC=$?
kill $P_SMI $P_DMON $P_CPU 2>/dev/null; [ -n "$P_SPY" ] && wait $P_SPY 2>/dev/null
echo "$(date -u +%FT%TZ) W2perf_$MODE engine rc $RC" | tee -a $OUT/READ_head.txt
awk -F', ' 'NR > 1 && $2+0 == $2+0 { n++; u += $2; if ($2 + 0 < 10) lo++; if ($2 + 0 > 90) hi++; m = ($4 + 0 > m) ? $4 + 0 : m } END { if (n) printf "telemetry: %d samples, mean utilization.gpu %.1f %%, below 10 %% %.1f %%, above 90 %% %.1f %%, memory.used max %d MiB\n", n, u / n, 100 * lo / n, 100 * hi / n, m }' $OUT/smi.csv | tee -a $OUT/READ_head.txt
exit $RC
