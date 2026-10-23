#!/bin/bash
# Strand 5 J5 decomposition of spec 10(p'''''), rerun with every channel (strand5_w0radial_v10.py --jmax-factor 4): the builder's angular-resolution
# control J5jres1 showed that at the j count n the Gaussians at R_max = 4.5 w lose about 1.4 percent of W_0^(2), the two-width k 1.3 percent, and c_3
# is wrong; at 2n and 4n they agree to about 3e-4.  The fast route, 8 workers (F'''(d)'s entropies on the other 8).  Gaussians psi = P + e f_w, N = 1,
# e = +-1e-2, R_max = 4.5 (grid) w, the j count 4n:
#   (1) h = 1, rulers P = 1, 1.05, 1.1, 1.2, frame widths 30 and 100 (grid widths w/P^2);  (2) h = 1.44, P = 1, w = 30 and 100, e = +-1e-2 and
#   +-1e-2/1.2 (the identity check against (1)'s P = 1.2);  (3) h = 0.5 as (1);  (4) h = 0.25, P = 1, w = 30 and 100;  (5) h = 1, P = 1, w = 300.
# Version 2's (6), w = 300 at h = 0.5 (about 2.5 hours at 4n), is not queued: at 4n the width pairs (20, 60) and (30, 100) give k within 3.3e-4 of
# each other (J5jres1), and (5) measures the q^6 term at h = 1; the lead may re-queue it.  Starts at once.  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
O=J5decomp3
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
true
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
run() {  # tag h rulers widths frame [gauss-eps]
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v10.py \
    --family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz --zc 0.823379 --h $2 \
    --rmax-over-rs 4 --what gauss --eps "" --gauss-eps ${6:-1e-2} --rulers $3 --widths $4 --width-frame $5 --gauss-rmax-over-w 4.5 --gpu -1 \
    --route fast --workers 8 --nt 120 --jmax-factor 4 --out $O --tag $1 > $O/$1.log 2>&1
  echo "$1 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
run gauss_h1 1 1,1.05,1.1,1.2 30,100 scaled
run gauss_h1.44 1.44 1 30,100 grid 1e-2,8.333333333333334e-3
run gauss_h0.5 0.5 1,1.05,1.1,1.2 30,100 scaled
run gauss_h0.25 0.25 1 30,100 grid
run gauss_h1_w300 1 1 300 grid
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
