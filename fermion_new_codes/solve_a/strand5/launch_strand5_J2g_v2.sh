#!/bin/bash
# Strand 5 J2g (strand5_J2g_v2.py; spec sec. 10(q)): the physical inventory's ruler derivative and eq. 7's piece on J1's profiles; builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
mkdir -p J2g_run2
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > J2g_run2/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand5_J2g_v2.py --out J2g_run2 --workers 4 > J2g_run2/run.log 2>&1
echo "exit $? $(date -u '+%H:%M:%S')" >> J2g_run2/status.txt
