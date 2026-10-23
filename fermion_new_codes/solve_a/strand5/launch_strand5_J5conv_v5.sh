#!/bin/bash
# Strand 5 J5, spec 10(p'''''') (2) and (3) at h = 1 (slot A, 8 workers; strand5_w0radial_v10.py, the fast route, nt 200, the j count n, which is
# complete inside R_max/4 = r_s): (2) the pilot ((0.1, 0.5) kernel g = 0, z_c 0.823379) at h = 0.125, R_max = 4 r_s: the star at e = 1 and the
# exact-L pair at lam = 1, e = +-1e-2 (densities kept; table Q's d_lin from the pair); then (3) the second member, the (0.03, 0.3) kernel g = 0
# family's profile nearest its maximum (z_c 0.823379; the family's maximum is at 0.83448), at h = 1, R_max = 4 r_s, the same three configurations.
# Waits for F'''(d)'s report (thread budget: J5decomp3 holds the other 8).  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
mkdir -p J5conv5 J5member1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > J5conv5/status.txt
until grep -q "^report exit" ${QI2_ROOT}/status.txt; do sleep 30; done
run() {  # out tag family profiles h
  date -u "+$2 start %Y-%m-%d %H:%M:%S UTC" >> $1/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v10.py \
    --family $3 --profiles $4 --zc 0.823379 --h $5 --rmax-over-rs 4 --what star,lin --eps "" --star-eps none --lin-eps 1e-2 --lin-eps-dil none \
    --jmax-factor 1 --gpu -1 --route fast --workers 8 --nt 200 --out $1 --tag $2 > $1/$2.log 2>&1
  echo "$2 exit $? $(date -u +%H:%M:%S)" >> $1/status.txt
}
run J5conv5 star_h0.125_R4 J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz 0.125
echo DONE >> J5conv5/status.txt
run J5member1 star_m0.03_h1_R4 J4run2/lstar_m0.03_aH0.3_kernel_g0_0.json J4run2/lstar_m0.03_aH0.3_kernel_g0_0_profiles.npz 1
echo "slot A DONE" >> J5member1/status.txt
