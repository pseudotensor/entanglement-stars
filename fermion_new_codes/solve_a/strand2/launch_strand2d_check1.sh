#!/bin/bash
# Strand 2d library checks (strand2d_check_v1.py) on GPU 2 if it has no compute process at launch (GPU 1 is another user's); 2 threads.
set -u
D=${QI2_ROOT}/s2d_jet
cd $D || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
U2=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
if [ -n "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$U2")" ]; then echo "GPU 2 busy at launch" > check1.log; exit 1; fi
nice -n 10 ${QI2_ROOT}/python code/strand2d_check_v1.py --out check1 --device cuda:2 > check1.log 2>&1
echo "exit $?" >> check1.log
