#!/bin/bash
# pod 3 (inbox #77; replaces chain_wl_then_C_pod3_v1.sh, stopped by its PID): C's sigma/a 4 member starts as soon as the wavelength set's last GPU
# run has ended (the 8 component runs logged), not after the set's CPU-only analysis, so the card does not wait for the analysis; the analysis
# runs beside C on the CPU (208 cores). Everything on the container disk.
ST=${QI2_ROOT}/wl_status.txt; B=${QI2_ROOT}/busi_compC
until [ "$(grep -cE 'matched in (above|below) exit' $ST)" -ge 8 ]; do sleep 30; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 15; done
echo "chain v2: the wavelength set's GPU runs done, C32_s4 starting $(date -u +%FT%TZ)" >> $B/status.txt
(setsid nohup bash $B/code/pod_launch_C_v2.sh s4 $B > $B/launch.out 2>&1 < /dev/null &)
sleep 20
(setsid nohup bash $B/code/pod_guard_C_v2.sh s4 $B > /dev/null 2>&1 < /dev/null &)
