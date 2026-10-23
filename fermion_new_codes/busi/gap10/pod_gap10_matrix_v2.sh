#!/bin/bash
# GAP10 (B5) matrix, RUN_ORDER_v2 section 2 (builder): T1 to T6 one process at a time on the pod's GPU, started when J4's
# stage E (the higher-ranked B4) has ended, then the analysis of section 4. T7 waits for the record table (section 3).
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
grep -A1 '^# ---- analysis' $G/matrix.sh | tail -1 | bash > $G/analysis.log 2>&1; echo "analysis exit $? $(date -u +%H:%M:%S)" >> $ST
echo "all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
