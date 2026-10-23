#!/bin/bash
# Strand 2c addendum parts B and C (strand2c_addendum_BC_v1.py) on GPU 2, after part A's kernel (kernel1) has finished and GPU 2 has no
# compute process; the ring4 P2 profiles of run3; 2 threads.
set -u
D=${QI2_ROOT}/s2c_mismatch
cd $D || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p addBC1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > addBC1/status.txt
until grep -q "kernel exit" kernel1/status.txt 2>/dev/null; do sleep 30; done
U2=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
until [ -z "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$U2")" ]; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC (GPU 2 free)' >> addBC1/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand2c_addendum_BC_v1.py --out addBC1 --profiles run3/p2_profiles_R4.json --kernel kernel1/kernel_L256.json --gpu 2 > addBC1/addBC.log 2>&1
echo "BC exit $?" >> addBC1/status.txt
echo DONE >> addBC1/status.txt
