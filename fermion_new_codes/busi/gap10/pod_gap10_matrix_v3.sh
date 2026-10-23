#!/bin/bash
# GAP10 (B5) matrix, RUN_ORDER_v2 sections 2 to 4 (builder): T1 to T6 one process at a time on the pod's GPU, started when
# J4's stage E (the higher-ranked B4) has ended; then T7 on the solve lane's r_s/a 32 slice at T/r_s 18 (busi/notes/
# GAP10_T7_record_plan_v1.md: the 4 runs and their 4 matched-preparation twins); then the analysis. Replaces v2 (no T7).
cd ${QI2_ROOT}/gap10
G=${QI2_ROOT}/gap10; ST=$G/matrix_status.txt; SS=${QI2_ROOT}/v2_stageDE_status.txt
until grep -q "stageE production exit" $SS 2>/dev/null; do sleep 120; done
echo "matrix start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
for T in T1 T2 T3 T4 T5 T6; do
  grep -v '^#' $G/matrix.sh | grep gap10_run_v2 | grep "/$T" | while read -r c; do
    bash -c "$c" >> $G/logs_$T.log 2>&1; echo "$T run exit $? $(date -u +%H:%M:%S)" >> $ST
  done
  echo "$T done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
done
B="python ${QI2_ROOT}/gap10_run_v2.py --rs 29 --ma 0.2 --ell 4 --Rb_frac 3.86 --profile file --profile_file $G/records/record_slice.npz --sampling point --source both --out $G/runs/T7_record --inside_twins_max_depth -1"
for EXTRA in "" "--prep matched --tag _matched"; do
  for DD in "100 2" "100 8" "111 2" "111 8"; do
    set -- $DD
    $B --direction $1 --depth $2 $EXTRA >> $G/logs_T7.log 2>&1; echo "T7 run $1 d$2 ${EXTRA:+matched} exit $? $(date -u +%H:%M:%S)" >> $ST
  done
done
echo "T7 done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
grep -A1 '^# ---- analysis' $G/matrix.sh | tail -1 | bash > $G/analysis.log 2>&1; echo "analysis exit $? $(date -u +%H:%M:%S)" >> $ST
echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
