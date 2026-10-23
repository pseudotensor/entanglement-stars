#!/bin/bash
# Strand 5 J5 decomposition, redesigned (spec 10(p'''''); strand5_w0radial_v9.py, the fast route, 8 workers; the dense GPU route is not used: at the
# grids here, n up to 2700, its cost n^4 per configuration is out of reach).  Gaussians psi = P + e f_w, N = 1, e = +-1e-2, R_max = 4.5 (grid) w:
#   (1) h = 1, rulers P = 1, 1.05, 1.1, 1.2, frame widths 30 and 100 (grid widths w/P^2: the same widths in the flat frame at grid spacing P^2 h);
#   (2) h = 1.44, P = 1, w = 30 and 100, e = +-1e-2 and +-1e-2/1.2: the identity check against (1)'s P = 1.2 (same flat-frame widths;
#       config by config W(1.2, e; h = 1, w/1.44) = W(1, e/1.2; h = 1.44, w), checked at h = 2 against 2.88 to 9e-8 in the smoke test);
#   (3) h = 0.5, as (1);  (4) h = 0.25, P = 1 only, w = 30 and 100;  (5) h = 1, P = 1, w = 300;  (6) h = 0.5, P = 1, w = 300 (the three-width fit).
# Waits for J5conv4 (thread budget; F'''(d)'s entropy step runs beside it on 8 workers).  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
O=J5decomp2
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until grep -q DONE J5conv4/status.txt; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
run() {  # tag h rulers widths frame [gauss-eps]
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v9.py \
    --family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz --zc 0.823379 --h $2 \
    --rmax-over-rs 4 --what gauss --eps "" --gauss-eps ${6:-1e-2} --rulers $3 --widths $4 --width-frame $5 --gauss-rmax-over-w 4.5 --gpu -1 \
    --route fast --workers 8 --nt 120 --out $O --tag $1 > $O/$1.log 2>&1
  echo "$1 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
run gauss_h1 1 1,1.05,1.1,1.2 30,100 scaled
run gauss_h1.44 1.44 1 30,100 grid 1e-2,8.333333333333334e-3
run gauss_h0.5 0.5 1,1.05,1.1,1.2 30,100 scaled
run gauss_h0.25 0.25 1 30,100 grid
run gauss_h1_w300 1 1 300 grid
run gauss_h0.5_w300 0.5 1 300 grid
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
