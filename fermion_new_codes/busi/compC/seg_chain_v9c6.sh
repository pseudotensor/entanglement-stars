#!/bin/bash
# the RD-4 box segment arms in sequence on one GPU: seg_chain_v9c6.sh GPU [WAIT_PID] ARM ...  (GPU never 0; WAIT_PID: an own earlier arm to wait for, or -)
GPU=${1:?gpu}; WP=${2:?wait pid or -}; shift 2; ST=${QI2_ROOT}/v9c6
if [ "$WP" != "-" ]; then while kill -0 $WP 2>/dev/null; do sleep 20; done; echo "waited for $WP $(date -u +%FT%TZ)"; fi
for arm in "$@"; do bash $ST/run_seg_v9c6.sh $arm $GPU; done
echo "chain done $(date -u +%FT%TZ)"
