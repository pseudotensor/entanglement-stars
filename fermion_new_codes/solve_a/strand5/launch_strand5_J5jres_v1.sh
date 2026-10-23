#!/bin/bash
# Strand 5 J5 angular-resolution control (builder's addition, revision 3; strand5_w0radial_v10.py, the fast route, 8 workers): versions 4 to 9 keep
# j < n (n = R_max/h), so beyond r = R_max/4 the channels between n and 4 r/h are absent (channel start floor((j + 1/2)/4)).  At h = 1, P = 1:
#   (1) the Gaussians w = 20, 30, 60, 100 at R_max = 4.5 w with the j count 4n (all channels at every radius), e = +-1e-2;
#   (2) w = 30 and 100 at R_max = 4.5 w with 2n;  (3) w = 30 at R_max = 8 w with 4n (the wall at fixed full resolution);
#   (4) the pilot star (R_max = 4 r_s) at e = 1 and +-1e-2 with the densities, j count 2n and 4n (table Q's exterior at full resolution).
# Waits for J5decomp2's running h = 0.5 scan (its launcher was stopped; thread budget).  Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
O=J5jres1
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
while ps -u jon -o args= | grep -qE "[s]trand5_w0radial_v9.py.*--tag gauss_h0.5$"; do sleep 20; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
run() {  # tag what widths rmax-over-w jfac star-eps nt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v10.py \
    --family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz --zc 0.823379 --h 1 \
    --rmax-over-rs 4 --what $2 --eps "" --gauss-eps 1e-2 --star-eps $6 --rulers 1 --widths $3 --width-frame grid --gauss-rmax-over-w $4 \
    --jmax-factor $5 --gpu -1 --route fast --workers 8 --nt $7 --out $O --tag $1 > $O/$1.log 2>&1
  echo "$1 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
run gauss_h1_R4.5_j4 gauss 20,30,60,100 4.5 4 none 120
run gauss_h1_R4.5_j2 gauss 30,100 4.5 2 none 120
run gauss_h1_w30_R8_j4 gauss 30 8 4 none 120
run star_h1_R4_j2 star 30 4.5 2 1e-2 200
run star_h1_R4_j4 star 30 4.5 4 1e-2 200
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
