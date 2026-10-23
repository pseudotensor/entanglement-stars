#!/bin/bash
# runs gate F only after gate D version 2 has ended with "GATE D: PASS", at a 1-minute load below 90 (the remote-box rule).
R=${QI2_ROOT}/compC; C=${QI2_ROOT}/compC
until grep -q "gateD2 exit" $R/status.txt; do sleep 60; done
if ! grep -q "^GATE D: PASS" $R/gateD_v2.log; then echo "gateF_s4 not run: gate D v2 did not pass $(date -u +%H:%M:%S)" >> $R/status.txt; exit 0; fi
until awk '{exit !($1 < 90)}' /proc/loadavg; do sleep 120; done
bash $C/gateF_box_v1.sh
