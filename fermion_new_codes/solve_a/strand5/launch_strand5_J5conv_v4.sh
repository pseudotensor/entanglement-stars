#!/bin/bash
# Strand 5 J5 pilot convergence in h (revision 2's finding: inside the star the nonlinear/linear ratio - 1 is 1.555e-2 at h = 1 and 9.99e-3 at
# h = 0.5): h = 0.25, R_max = 4 r_s, the fast route (12 workers), the star at e = 1 and +-1e-2 (the densities) and the exact-L set at lam = 1,
# e = +-1e-2. Waits for J5conv3 (thread budget). Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
O=J5conv4
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until grep -q DONE J5conv3/status.txt; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v7.py --family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz --zc 0.823379 --h 0.25 --rmax-over-rs 4 --what star,lin --eps "" --star-eps 1e-2 --lin-eps 1e-2 --lin-eps-dil none --gpu -1 --route fast --workers 12 --nt 200 --out $O --tag star_h0.25_R4 > $O/star_h0.25_R4.log 2>&1
echo "star_h0.25_R4 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
echo DONE >> $O/status.txt
