#!/bin/bash
# Strand 5 J5 control (builder's addition, revision 3): the Gaussians' dependence on the domain at h = 1, P = 1, e = +-1e-2 (strand5_w0radial_v9.py,
# the fast route, 8 workers).  The dense pilot's Gaussians (w = 20 at R_max = 322 = 16.1 w, w = 60 at R_max = 480 = 8 w) give c_3 = -0.0541 and
# -0.0545, J5decomp2's (R_max = 4.5 w) -0.0262 (w = 30) and +0.0006 (w = 100); the two-width k differs by 1.3 percent between the width pairs
# (20, 60) and (30, 100).  Each width at three domains: w = 20 at R_max/w = 4.5 and 8; w = 30 at 8 and 16; w = 60 at 4.5 and 16; w = 100 at 8
# (the others exist).  Waits for F'''(d)'s entropy step (thread budget: J5decomp2 runs on 8 workers).  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
O=J5domain1
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until grep -q DONE ${QI2_ROOT}/status.txt; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
run() {  # tag widths rmax-over-w
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v9.py \
    --family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz --zc 0.823379 --h 1 \
    --rmax-over-rs 4 --what gauss --eps "" --gauss-eps 1e-2 --rulers 1 --widths $2 --width-frame grid --gauss-rmax-over-w $3 --gpu -1 \
    --route fast --workers 8 --nt 120 --out $O --tag $1 > $O/$1.log 2>&1
  echo "$1 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
run gauss_h1_w20_R4.5 20 4.5
run gauss_h1_w20_R8 20 8
run gauss_h1_w30_R8 30 8
run gauss_h1_w30_R16 30 16
run gauss_h1_w60_R4.5 60 4.5
run gauss_h1_w60_R16 60 16
run gauss_h1_w100_R8 100 8
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
