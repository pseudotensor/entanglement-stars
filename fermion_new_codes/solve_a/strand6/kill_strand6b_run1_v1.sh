#!/bin/bash
# Stop this lane's strand 6 (b) CPU run (s6_validation/omega/run1): the launcher shell and its python processes, by PID (no pkill -f).
for p in $(ps -u jon -o pid=,args= | awk '/launch_strand6b_omega_v1.sh/ && !/awk/ {print $1}'); do kill $p 2>/dev/null && echo "killed launcher $p"; done
for p in $(ps -u jon -o pid=,args= | awk '/strand6b_omega_v2.py/ && /omega\/run1/ && !/awk/ {print $1}'); do kill $p 2>/dev/null && echo "killed $p"; done
sleep 2
ps -u jon -o pid=,args= | awk '/strand6b_omega_v2.py/ && !/awk/' | wc -l
