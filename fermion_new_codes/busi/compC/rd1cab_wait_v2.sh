#!/bin/bash
# busi, PERF RD-1c: wait (15 s polls) until the h100-f GPU holds no compute process, then the RD-1c A/B (rd1cab_run_all_v2.sh, which waits
# the same way before each arm). v1 waited for 3 free checks 60 s apart, which MERGE-1's back-to-back arms (60 s polls) never leave.
BASE=${QI2_ROOT}/busi_rd1c
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 15; done
echo "GPU free $(date -u +%FT%TZ)"; bash $BASE/rd1cab_run_all_v2.sh
