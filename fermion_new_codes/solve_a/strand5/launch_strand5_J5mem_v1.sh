#!/bin/bash
# Strand 5 J5, spec 10(p'''''') (3) at h = 0.5 (slot B; strand5_w0radial_v10.py, the fast route, nt 200, 8 workers): J5mem2, the (0.03, 0.3) kernel
# g = 0 family's member nearest its maximum (z_c 0.823379) at h = 0.5, R_max = 4 r_s (n = 4528), the j count n: the flat control, the star at e = 1
# (densities kept) and the exact-L pair at lam = 1, e = +-1e-2 (densities kept).  About 7 hours on 8 workers.  Waits for J5decomp3 (thread budget).
# Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
mkdir -p J5mem2
until grep -q DONE J5decomp3/status.txt; do sleep 30; done
date -u "+star_m0.03_h0.5_R4 start %Y-%m-%d %H:%M:%S UTC" >> J5mem2/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v10.py \
  --family J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json --profiles J4run2/lstar_m0.03_aH0.3_kernel_g0_0_profiles.npz --zc 0.823379 --h 0.5 \
  --rmax-over-rs 4 --what flat,star,lin --eps "" --star-eps none --lin-eps 1e-2 --lin-eps-dil none --jmax-factor 1 --gpu -1 --route fast --workers 8 \
  --nt 200 --out J5mem2 --tag star_m0.03_h0.5_R4 > J5mem2/star_m0.03_h0.5_R4.log 2>&1
echo "star_m0.03_h0.5_R4 exit $? $(date -u +%H:%M:%S)" >> J5mem2/status.txt
echo "slot B DONE" >> J5mem2/status.txt
