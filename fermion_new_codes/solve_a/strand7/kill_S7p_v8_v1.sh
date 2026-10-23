#!/bin/bash
# Stops launch_strand7_S7p_v8.sh (the launcher and its two slot subshells, which have not started a job) by PID before any job starts; ps/awk
# only.  Version 9 replaces it with GPU 1's slot gated on H's start (version 7's slot claims GPU 1 when K1 ends).  Builder analyst.
cd ${QI2_ROOT}/s7_collapse || exit 1
if grep -q " start .* (v6 eigh)" S7p_run3/status.txt; then echo "a v8 job has started; not killing"; exit 1; fi
for pid in $(ps -eo pid,args | awk '$2=="bash" && $3=="code/launch_strand7_S7p_v8.sh" {print $1}'); do echo "kill $pid"; kill $pid; done
