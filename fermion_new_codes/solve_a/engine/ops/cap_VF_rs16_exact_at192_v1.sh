#!/bin/bash
# The J-VF cap of SPEC_final4b_engine.md sec. 15.2 (1): "r_s = 16 exact massless (running) to kappa = 192 and stopped". Waits until the running
# line's log prints its kappa = 192 line (the record is dumped at that line), then stops that python process (exact-argument match) and its
# list runner (so the v2 list's next line, no-trap to 512, does not start); the re-planned lines run from list_VF_v3.txt. Builder analyst.
LOG=${QI2_ROOT}/rs16_exact_m0_v2.log
until grep -q "^kappa 192 (" $LOG; do sleep 15; done
sleep 5
pids=$(ps -eo pid=,args= | awk '$3=="code/engine_vacflow_v2.py" {for (i=1;i<=NF;i++) if ($i=="--tag" && $(i+1)=="rs16_exact_m0_v2") print $1}')
rpids=$(ps -eo pid=,args= | awk '$2=="bash" && $3=="code/launch_engine_VF_v2.sh" {print $1}')
for p in $rpids; do kill -TERM $p && echo "J-VF runner v2 (pid $p) stopped $(date -u +%H:%M:%S) UTC (the re-plan of engine 15.2)" >> ${QI2_ROOT}/status.txt; done
for p in $pids; do kill -TERM $p && echo "J-VF rs16_exact_m0_v2 (pid $p) capped at kappa = 192 $(date -u +%H:%M:%S) UTC (engine 15.2 (1)); its record dumped at kappa 192" >> ${QI2_ROOT}/status.txt; done
