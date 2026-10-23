#!/bin/bash
# after the A1 stall (box contention, no band-gather samples): B8 then AB8 on the segment, one at a time
cd ${QI2_ROOT}/H3_build
for arm in B8 AB8; do echo "load before $arm: $(cut -d" " -f1-3 /proc/loadavg)"; bash ${QI2_ROOT}/run_h3seg_v9c4.sh $arm; done
echo "chain2 done $(date -u +%FT%TZ)"
