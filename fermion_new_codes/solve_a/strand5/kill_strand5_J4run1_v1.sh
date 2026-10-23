#!/bin/bash
# Stop this lane's J4 stage-1 queue J4run1 (superseded by version 2 with profiles): the launcher and its python process, by PID.
for p in $(ps -u jon -o pid=,args= | awk '/launch_strand5_J4_v1.sh/ && !/awk/ {print $1}'); do kill $p 2>/dev/null && echo "killed launcher $p"; done
for p in $(ps -u jon -o pid=,args= | awk '/strand5_lattice_star_v1.py/ && /J4run1/ && !/awk/ {print $1}'); do kill $p 2>/dev/null && echo "killed $p"; done
sleep 1
ps -u jon -o pid=,args= | awk '/strand5_lattice_star_v1.py/ && !/awk/' | wc -l
