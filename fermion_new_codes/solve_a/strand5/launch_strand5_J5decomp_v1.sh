#!/bin/bash
# Strand 5 J5 decomposition (spec 10(p'''') (3); strand5_w0radial_v8.py, the fast route, 8 workers): the radial operator's own k_a(P) at uniform
# rulers P = 1, 1.05, 1.1, 1.2 from Gaussians psi = P + e f_w (w = 30 and 100 at m a = 0.1; e = +-1e-2; R_max = 4.5 w), at h = 1, 0.5, 0.25.
# Waits for J5conv4 (thread budget; F'''(d)'s entropy step runs beside it on 8 workers). Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
O=J5decomp1
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until grep -q DONE J5conv4/status.txt; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
for h in 1 0.5 0.25; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand5_w0radial_v8.py --family J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json --profiles J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz --zc 0.823379 --h $h --rmax-over-rs 4 --what gauss --eps "" --gauss-eps 1e-2 --rulers 1,1.05,1.1,1.2 --widths 30,100 --gauss-rmax-over-w 4.5 --gpu -1 --route fast --workers 8 --nt 120 --out $O --tag gauss_h$h > $O/gauss_h$h.log 2>&1
  echo "gauss_h$h exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
