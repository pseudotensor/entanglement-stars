#!/bin/bash
# busi f52b1152: wait until the h100-f GPU holds no compute process and no SYM-W1 replay script runs (3 checks 60 s apart), then the SYM A/B.
BASE=${QI2_ROOT}/busi_compC; ok=0
while [ $ok -lt 3 ]; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); other=$(pgrep -u qi2 -f "[r]eplay_symw1|[b]usi_symw1/code" | head -1)
  if [ -z "$apps" ] && [ -z "$other" ]; then ok=$((ok+1)); else ok=0; fi; sleep 60
done
echo "GPU free $(date -u +%FT%TZ)"; bash $BASE/symab_run_all_v1.sh
