#!/bin/bash
# Strand 2d on B4 (strand2d_jet_v1.py): the vacuum jet (H = 1 and 0.5; CPU) and the matter jets at (g_M, g_B) = (0, 0) and (0.2, 1),
# Lambda 1, m 0.1, q in {0, pi/2, pi}, h in {1e-3, 1e-4}, on GPU 2, launched only if GPU 2 has no compute process (GPU 1 is another
# user's; GPU 0 holds 2c run3 until it ends).  2 threads.
set -u
D=${QI2_ROOT}/s2d_jet
cd $D || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p B4run1
U2=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
if [ -n "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$U2")" ]; then echo "GPU 2 busy at launch" > B4run1/status.txt; exit 1; fi
date -u '+start %Y-%m-%d %H:%M:%S UTC' > B4run1/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand2d_jet_v1.py --out B4run1 --lattice B4 --Hs 1,0.5 --couplings "0,0;0.2,1" --device cuda:2 > B4run1/jet.log 2>&1
echo "jet exit $?" >> B4run1/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> B4run1/status.txt
echo DONE >> B4run1/status.txt
