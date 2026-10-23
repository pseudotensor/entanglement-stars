#!/bin/bash
# Stops the version 1 repair chain (launch_strand7_repair_v1.sh and its current strand7_page_repair_v1.py step) by PID; version 2 replaces it
# (the classification of the unresolved pairs by X's singular values).  ps/awk only, no pattern kill.  Builder analyst.
for pid in $(ps -eo pid,args | awk '$2=="bash" && $3=="code/launch_strand7_repair_v1.sh" {print $1}'); do echo "kill chain $pid"; kill $pid; done
for pid in $(ps -eo pid,args | awk '$3=="code/strand7_page_repair_v1.py" {print $1}'); do echo "kill step $pid"; kill $pid; done
