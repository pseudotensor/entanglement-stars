#!/bin/bash
# Item 5's figure data (strand5_item5_data_v1.py, seconds, one thread) when the lane runs at most 15 busy processes.  Builder analyst.
D=${QI2_ROOT}/s5_star
cd $D || exit 1
mkdir -p item5_fig
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > item5_fig/status.txt
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 15 ]; do sleep 20; done
date -u '+start %H:%M:%S UTC' >> item5_fig/status.txt
OMP_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_item5_data_v1.py --out item5_fig/strand5_item5_fig_data.npz --caption item5_fig/caption_numbers.txt > item5_fig/data.log 2>&1
echo "data exit $? $(date -u +%H:%M:%S)" >> item5_fig/status.txt
