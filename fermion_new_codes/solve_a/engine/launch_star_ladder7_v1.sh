#!/bin/bash
# The (0.05, 1/4) ladder beyond its continuum N_B,max (8921.6) per engine spec 26.4 (4), on box GPU 2 after the running (0.1, 1/64) member
# (PID given) ends: star_ladder_v1 with the ell control at every member and the stop rules. Builder analyst.
cd ${QI2_ROOT}/engine
AFTER=$1
while kill -0 $AFTER 2>/dev/null; do sleep 60; done
echo "GPU2 star ladder7 start $(date -u +%H:%M:%S) UTC (star_ladder_v1, gpu 2)" >> status.txt
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 ${QI2_ROOT}/python code/star_ladder_v1.py --m 0.05 --g 0.25 --ell 8 --nsites 3072 \
  --NB-list 13382.427,17843.235,20519.721,23196.206,26764.853,31225.662,35686.471,41039.441,47284.574,53529.706,62451.324,71372.941,89216.177 \
  --out Sfam7 --tag-prefix ladder7 --status SfamSum/ladder_status.txt > SfamSum/log_ladder7.log 2>&1
echo "GPU2 star ladder7 exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
