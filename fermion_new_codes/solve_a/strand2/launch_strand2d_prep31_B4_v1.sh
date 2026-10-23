#!/bin/bash
# Strand 2d sec. 7(2), stage 1 on B4 (strand2d_prep31_v1.py): the free (3,1)' term of J^prep, the fixed-vacuum cost and the norm-defect
# check at (0, 0) and (0.2, 1), Lambda 1, m 0.1, on GPU 2 after the B4 jet run (B4run1) is DONE and GPU 2 has no compute process.
set -u
D=${QI2_ROOT}/s2d_jet
cd $D || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p prep31_B4
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > prep31_B4/status.txt
until grep -q DONE B4run1/status.txt 2>/dev/null; do sleep 30; done
U2=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
until [ -z "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$U2")" ]; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC (GPU 2 free)' >> prep31_B4/status.txt
for g in "0 0" "0.2 1"; do
  set -- $g
  nice -n 10 ${QI2_ROOT}/python code/strand2d_prep31_v1.py --out prep31_B4 --lattice B4 --gM $1 --gB $2 --device cuda:2 >> prep31_B4/run.log 2>&1
  echo "B4 ($1, $2) exit $?" >> prep31_B4/status.txt
done
echo DONE >> prep31_B4/status.txt
