#!/bin/bash
# Strand 0: regenerate the tables with report v3 (C10 line labelled by mass) from the full_v2 JSON outputs.
cd ${QI2_ROOT}/s0_source_allocation || exit 1
OUT=full_v2
nice -n 10 ${QI2_ROOT}/python code/strand0_report_v3.py --partA $OUT/partA.json --ringL $OUT/ringL.json --cube $OUT/cube.json --md $OUT/RESULT_strand0_tables_v3.md > $OUT/report_v3.log 2>&1
echo "report_v3 exit $?" >> $OUT/status.txt
