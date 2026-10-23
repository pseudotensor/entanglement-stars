#!/bin/bash
# Strand 5 J5mem2, the (0.03, 0.3) member at h = 0.5 (JUDGMENT_strand5_w0check.md sec. 6 (g): "h = 0.5 last"), restarted after the builder's stop
# at 22:25 UTC (the threads went to strand 7): the same settings as launch_strand5_J5mem_v1.sh (strand5_w0radial_v10.py, the member's h = 1 run's
# version) with 6 workers (the lane's thread budget), started when the lane runs at most 10 busy processes.  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 10 ]; do sleep 30; done
date -u "+star_m0.03_h0.5_R4 restart %Y-%m-%d %H:%M:%S UTC (6 workers)" >> J5mem2/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v10.py \
  --family J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json --profiles J4run2/lstar_m0.03_aH0.3_kernel_g0_0_profiles.npz --zc 0.823379 --h 0.5 \
  --rmax-over-rs 4 --what flat,star,lin --eps "" --star-eps none --lin-eps 1e-2 --lin-eps-dil none --jmax-factor 1 --gpu -1 --route fast --workers 6 \
  --nt 200 --out J5mem2 --tag star_m0.03_h0.5_R4 > J5mem2/star_m0.03_h0.5_R4.log 2>&1
echo "star_m0.03_h0.5_R4 exit $? $(date -u +%H:%M:%S)" >> J5mem2/status.txt
echo "h0.5 DONE" >> J5mem2/status.txt
