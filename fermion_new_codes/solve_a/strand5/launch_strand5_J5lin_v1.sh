#!/bin/bash
# Strand 5 J5 pilot, the exact-L linear response (strand5_w0radial_v6.py --what lin, the dense route on GPU 2): the pilot member, h = 1,
# R_max = 4 r_s, lam = 1 and 1 -+ 1e-3, 1 -+ 2e-3, e = +-1e-2 and +-2e-2. One compute process on the GPU, nice 10, 2 threads. Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
O=J5pilot2
echo "lin start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" >> $O/status.txt
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v6.py --family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz --zc 0.823379 --h 1 --rmax-over-rs 4 --what lin --lin-eps 1e-2,2e-2 --gpu 0 --route dense --out $O --tag lin_h1_R4 > $O/lin_h1_R4.log 2>&1
echo "lin_h1_R4 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
