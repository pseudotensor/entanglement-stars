#!/bin/bash
# Strand 4 Part B, the lead's audit after the reviewer B red team (strand4_partB_audit_v1.py: items 2 and 4), one thread; then the Part A
# tables regenerated with the kernel-normalized screening length (strand4_partA_report_v2.py) for partA_run2_early and partA_run1.
set -u
D=${QI2_ROOT}/s4_controls
cd $D || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p partB_audit1
date -u '+start %Y-%m-%d %H:%M:%S UTC' > partB_audit1/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand4_partB_audit_v1.py --inputs $D/partB_inputs --out partB_audit1 --stage2-members $D/partB_inputs/dev/stage2_members.json > partB_audit1/audit.log 2>&1
echo "audit exit $?" >> partB_audit1/status.txt
for R in partA_run2_early partA_run1; do
  nice -n 10 ${QI2_ROOT}/python code/strand4_partA_report_v2.py --run $R --md $R/RESULT_strand4_partA_tables_v2.md > $R/report_v2.log 2>&1
  echo "partA report v2 $R exit $?" >> partB_audit1/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> partB_audit1/status.txt
echo DONE >> partB_audit1/status.txt
