#!/bin/bash
# Strand 2d sec. 10 item 3: J_NN by the exact sum over states at the dressed B4 flat point (strand2d_jnn_v1.py), after the sec. 9 rerun
# (B4run2) in the same two-thread slot, on any GPU free at launch.  Output s2d_jet/jnn1.
set -u
B=${QI2_ROOT}/s2d_jet
cd $B || exit 1
mkdir -p jnn1
date -u '+queued (after B4run2) %Y-%m-%d %H:%M:%S UTC' > jnn1/status.txt
until grep -q DONE B4run2/status.txt 2>/dev/null; do sleep 30; done
gpu_wait() {
  while true; do
    BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
    for i in 0 1 2; do
      Ui=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i=$i '$1==i{print $2}')
      if [ -n "$Ui" ] && ! echo "$BUSY" | grep -q "$Ui"; then GPU=$i; return 0; fi
    done
    sleep 30
  done
}
gpu_wait
date -u "+start %Y-%m-%d %H:%M:%S UTC (GPU $GPU free)" >> jnn1/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand2d_jnn_v1.py --out jnn1 --device cuda:$GPU > jnn1/jnn.log 2>&1
echo "jnn exit $?" >> jnn1/status.txt
echo DONE >> jnn1/status.txt
