#!/bin/bash
# busi, PERF RD-1c: wait until MERGE-1's h100-f chain has written "chain ALL DONE" (or the GPU has held no compute process for 10 checks 60 s
# apart), then until the GPU holds no compute process (3 checks 60 s apart), then the RD-1c A/B.
BASE=${QI2_ROOT}/busi_rd1c; MC=${QI2_ROOT}/chain.log; idle=0
until grep -q "chain ALL DONE" $MC 2>/dev/null || [ $idle -ge 10 ]; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); if [ -z "$apps" ]; then idle=$((idle+1)); else idle=0; fi; sleep 60
done
echo "merge chain: $(tail -1 $MC 2>/dev/null) (idle checks $idle) $(date -u +%FT%TZ)"
ok=0
while [ $ok -lt 3 ]; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  if [ -z "$apps" ]; then ok=$((ok+1)); else ok=0; fi; sleep 60
done
echo "GPU free $(date -u +%FT%TZ)"; bash $BASE/rd1cab_run_all_v1.sh
