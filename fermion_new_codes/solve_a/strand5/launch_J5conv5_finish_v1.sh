#!/bin/bash
# Writes J5conv5's exit and DONE lines (slot A's launcher was stopped while star_h0.125_R4 ran) when the run's log ends in "done" or an error.
# Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
until tail -n 1 J5conv5/star_h0.125_R4.log 2>/dev/null | grep -q "^done" || grep -q "Traceback\|Error" J5conv5/star_h0.125_R4.log 2>/dev/null; do sleep 60; done
if tail -n 1 J5conv5/star_h0.125_R4.log | grep -q "^done"; then rc=0; else rc=1; fi
echo "star_h0.125_R4 exit $rc $(date -u +%H:%M:%S) (seen in its log; slot A's launcher was stopped)" >> J5conv5/status.txt
echo DONE >> J5conv5/status.txt
