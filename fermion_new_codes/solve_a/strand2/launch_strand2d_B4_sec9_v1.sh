#!/bin/bash
# Strand 2d, SPEC_strand2d_jet sec. 9 (builder analyst): the B4 (0.2, 1) jets at q = 0, pi/2, pi with h in {1e-4, 1e-5, 1e-6}, the
# lowest 125 levels at every point, the cluster average with d_cl = 120, the noise floor (strand2d_jet_v4.py, matter only).  Runs in slot D
# (two threads) after the 2a red-team checks end, on any GPU free at launch.  Output B4run2.
set -u
B=${QI2_ROOT}/solve_a
cd $B/s2d_jet || exit 1
mkdir -p B4run2
date -u '+queued (slot D, after the 2a checks) %Y-%m-%d %H:%M:%S UTC' > B4run2/status.txt
until grep -q DONE $B/s2a_gate/rtchecks1/status.txt 2>/dev/null; do sleep 30; done
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
date -u "+start %Y-%m-%d %H:%M:%S UTC (GPU $GPU free)" >> B4run2/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 /usr/bin/time -v ${QI2_ROOT}/python code/strand2d_jet_v4.py --out B4run2 --lattice B4 --couplings "0.2,1" --hs 1e-4,1e-5,1e-6 --qs "0,pi/2,pi" --nlevels 125 --d-cluster 120 --skip-vacuum --device cuda:$GPU > B4run2/jet.log 2>&1
echo "jet exit $?" >> B4run2/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> B4run2/status.txt
echo DONE >> B4run2/status.txt
