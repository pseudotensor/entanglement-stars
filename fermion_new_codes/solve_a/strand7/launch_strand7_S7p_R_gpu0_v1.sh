#!/bin/bash
# Item 7, R (the r_h/h scaling test: L at r_s0 = 64 h, R_max = 40 r_s0, j = 1/2) on GPU 0 (an RTX A6000, free at launch; the repair chain's
# next step waits for it) on strand7_sea_page_v6.py --evolution eigh, moved from version 9's queue to bring item 7's result forward; one CPU
# thread.  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
O=S7p_run3
gpu_free() { [ -z "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
until gpu_free 0 && [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 20; done
date -u "+R start %H:%M:%S on cuda:0 (v6 eigh)" >> $O/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand7_sea_page_v6.py --device cuda:0 --evolution eigh --background locked --rs0 64 \
  --Rmax-over-rs 40 --channels 0.5 --T-end-over-rs 40 --check-times 0,24,40 --out $O --tag R > $O/R.log 2>&1
echo "R exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
