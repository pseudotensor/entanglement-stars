#!/bin/bash
# pod 3 (inbox #75; turn 54 46.2 (5)): GAP10's wavelength set (pod_gap10_wl_v4.sh, about 1 h) finishes first, then Computation C's sigma/a 4
# member runs alone on the card (pod_launch_C_v2.sh s4 ${QI2_ROOT}/busi_compC) with its guard. Everything on the container disk; /workspace is
# not touched. The order: the shorter required job first costs C about 1 h, the reverse would cost GAP10 about 4 h.
ST=${QI2_ROOT}/wl_status.txt; B=${QI2_ROOT}/busi_compC
until grep -q "T7wl all done" $ST; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 30; done
echo "chain: wavelength set done, C32_s4 starting $(date -u +%FT%TZ)" >> $B/status.txt
(setsid nohup bash $B/code/pod_launch_C_v2.sh s4 $B > $B/launch.out 2>&1 < /dev/null &)
sleep 20
(setsid nohup bash $B/code/pod_guard_C_v2.sh s4 $B > /dev/null 2>&1 < /dev/null &)
