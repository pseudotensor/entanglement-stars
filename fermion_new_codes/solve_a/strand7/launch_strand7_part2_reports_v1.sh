#!/bin/bash
# S7a part 2's reports (SPEC_strand7_collapse.md secs. 10.2 and 11) once the dt/4 rung has landed: strand7_shell_report_v4.py over S7a_run2 and
# S7a_run3, then strand7_part2_report_v2.py (the three-rung ladder, F_disc on self E16's fields with the static calibration at the maximum
# member, the E-ladder, the moved exterior, the record checks), one CPU thread, sequential.  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=S7a_part2
date -u "+part 2 reports start %H:%M:%S" >> $O/status.txt
for r in S7a_run2 S7a_run3; do
  nice -n 10 $PY code/strand7_shell_report_v4.py --dir $r --frozen-ref S7a_run1/s7a_frozen_schw_E8_A.json --md $O/RESULT_shell_v4_$r.md --out-json $O/shell_v4_$r.json > $O/shell_v4_$r.log 2>&1
  echo "shell v4 $r exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
done
R2=S7a_run2; R3=S7a_run3
nice -n 10 $PY code/strand7_part2_report_v2.py --ladder $R2/s7a_self_E16_R4_A.json,$R2/s7a_self_E16_R4_A_dt.json,$R3/s7a_self_E16_R4_A_dt4.json \
  --fields $R2/s7a_self_E16_R4_A_fields.npz --e-ladder $R2/s7a_self_E8_R4_A.json,$R2/s7a_self_E16_R4_A.json,$R2/s7a_self_E32_R4_A.json \
  --rmax $R2/s7a_self_E16_R4_A.json:$R3/s7a_self_E16_R4_A_R128.json \
  --runs $R2/s7a_self_E8_R4_A.json,$R2/s7a_self_E16_R4_A.json,$R2/s7a_self_E16_R4_A_dt.json,$R2/s7a_self_E32_R4_A.json,$R2/s7a_self_E8_R4_Bp.json,$R3/s7a_self_E16_R4_A_dt4.json,$R3/s7a_self_E16_R4_A_R128.json,$R3/s7a_self_E8_R4_Bp2.json \
  --static S7a_run1/s7a_static_zc0.3.json,S7a_run1/s7a_static_zcmax.json --static-zc 0.83385376 \
  --frozen-pair S7a_run1/s7a_frozen_schw_E8_A.json:S7a_run1/s7a_frozen_schw_E8_A_verlet.json --md $O/RESULT_part2_report_v2.md --out-json $O/part2_report_v2.json > $O/part2_report_v2.log 2>&1
echo "part2 report v2 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
echo DONE >> $O/status.txt
