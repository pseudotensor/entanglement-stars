#!/bin/bash
# Strand 5 J5: version 5's fast route (CPU pool, 4 workers) on the pilot's setting (the (0.1, 0.5) kernel g = 0 member z_c = 0.823379, h = 1,
# R_max = 4 r_s): the flat control and the star series at e = 1 and +-1e-2, 2e-2, 4e-2, to compare with the dense route's records (J5pilot2);
# builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
O=J5fastcheck1
mkdir -p $O
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > $O/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v5.py --family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz --zc 0.823379 --h 1 --rmax-over-rs 4 --what flat,star --eps big --gpu -1 --route fast --workers 4 --nt 200 --out $O --tag star_h1_R4_fast > $O/star_h1_R4_fast.log 2>&1
echo "exit $? $(date -u '+%H:%M:%S')" >> $O/status.txt
