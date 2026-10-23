#!/bin/bash
# waits for gate C version 1's sigma/a 4 run to end, then (the box's 1-minute load below 90, the remote-box rule) gate C version 2 on both
# members in sequence: s4 then s2.
R=${QI2_ROOT}/compC; C=${QI2_ROOT}/compC
until grep -q "gateC_s4 exit" $R/status.txt; do sleep 60; done
for M in s4 s2; do
  until awk '{exit !($1 < 90)}' /proc/loadavg; do sleep 120; done
  bash $C/gateC_box_v2.sh $M
done
