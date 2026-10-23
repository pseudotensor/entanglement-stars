#!/bin/bash
# Strand 5 J5, spec 10(p'''''') (3) at h = 0.5 (slot B, 8 workers; strand5_w0radial_v10.py, the fast route, nt 200, the j count n): the second
# member ((0.03, 0.3) kernel g = 0, the profile at z_c 0.823379 nearest the family's maximum) at h = 0.5, R_max = 4 r_s (n = 4528): the star at
# e = 1 and the exact-L pair at lam = 1, e = +-1e-2 (densities kept).  About 5 hours.  Waits for J5decomp3 (thread budget).  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
mkdir -p J5member1
until grep -q DONE J5decomp3/status.txt; do sleep 30; done
date -u "+star_m0.03_h0.5_R4 start %Y-%m-%d %H:%M:%S UTC" >> J5member1/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v10.py \
  --family J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json --profiles J4run2/lstar_m0.03_aH0.3_kernel_g0_0_profiles.npz --zc 0.823379 --h 0.5 \
  --rmax-over-rs 4 --what star,lin --eps "" --star-eps none --lin-eps 1e-2 --lin-eps-dil none --jmax-factor 1 --gpu -1 --route fast --workers 8 \
  --nt 200 --out J5member1 --tag star_m0.03_h0.5_R4 > J5member1/star_m0.03_h0.5_R4.log 2>&1
echo "star_m0.03_h0.5_R4 exit $? $(date -u +%H:%M:%S)" >> J5member1/status.txt
echo "slot B DONE" >> J5member1/status.txt
