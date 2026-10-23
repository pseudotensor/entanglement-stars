#!/bin/bash
# busi, PERF RD-4: wait until the h100-f GPU holds no compute process (3 checks 60 s apart), then the RD-4 A/B.
BASE=${QI2_ROOT}/busi_rd4; ok=0
while [ $ok -lt 3 ]; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [ -z "$apps" ]; then ok=$((ok+1)); else ok=0; fi; sleep 60
done
echo "GPU free $(date -u +%FT%TZ)"; bash $BASE/rd4ab_run_all_v1.sh
