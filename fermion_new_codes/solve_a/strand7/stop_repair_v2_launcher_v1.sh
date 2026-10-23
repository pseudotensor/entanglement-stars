#!/bin/bash
# Final4: task 3's GPU runs come first. Stops the repair pass launcher (launch_strand7_repair_v2.sh) by PID so that it starts no further repair
# after the running one (K1 on cuda:0 finishes on its own; K2, R and H are left unrepaired: item 7's static-chart runs are diagnostics only).
# Builder analyst.
O=${QI2_ROOT}/S7p_run3
for pid in $(ps -eo pid,args | awk '/bash code\/launch_strand7_repair_v2\.sh/ && !/awk/ {print $1}'); do
  echo "stopping repair v2 launcher pid $pid: $(ps -o args= -p $pid)"
  kill $pid
done
date -u "+repair v2 launcher stopped %H:%M:%S (final4: task 3's GPU runs first; the running K1 repair finishes; K2, R not repaired)" >> $O/repair_status.txt
