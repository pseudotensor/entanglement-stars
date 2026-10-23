#!/bin/bash
# Strand 0 revision 2: regenerate the tables with report v5 (Part B heading corrected) from full_v3's JSON files.
cd ${QI2_ROOT}/s0_source_allocation || exit 1
OUT=full_v3
nice -n 10 ${QI2_ROOT}/python code/strand0_report_v5.py --partA $OUT/partA.json --ringL $OUT/ringL.json --cube $OUT/cube.json --md $OUT/RESULT_strand0_tables.md > $OUT/report_v5.log 2>&1
echo "report_v5 exit $?" >> $OUT/status.txt
