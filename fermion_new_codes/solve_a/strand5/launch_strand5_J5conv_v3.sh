#!/bin/bash
# Strand 5 J5 pilot convergence, launcher version 3 (strand5_w0radial_v6.py, the fast route, 12 CPU workers): at (h = 0.5, R_max = 4 r_s) and
# (h = 1, R_max = 8 r_s): the flat control, the star series (e = 1, +-1e-2, 2e-2, 4e-2), the nonlinear dilation family (lam = 1 -+ 1e-3,
# 1 -+ 2e-3; the quadratic-form dilation configurations skipped) and the exact-L linear response at e = +-1e-2 (lam = 1 and the dilation
# family); then the fast route's nt check (h = 1, R_max = 4 r_s, nt 300: e = 1 and the star series). Waits for the soft-direction run, the fast
# check and the second-crossing pass (thread budget). Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
O=J5conv3
mkdir -p $O
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until grep -q DONE J4softdir1/status.txt && grep -q exit J5fastcheck1/status.txt && grep -q DONE J4refine3/status.txt; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
PY=${QI2_ROOT}/python
F=J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json
P=J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz
run() {  # h rmax tag what nt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/strand5_w0radial_v6.py --family $F --profiles $P --zc 0.823379 --h $1 --rmax-over-rs $2 --what $4 --eps big --dil-eps none --lin-eps 1e-2 --gpu -1 --route fast --workers 12 --nt $5 --out $O --tag $3 > $O/$3.log 2>&1
  echo "$3 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt
}
run 0.5 4 star_h0.5_R4 flat,star,dil,lin 200
run 1 8 star_h1_R8 flat,star,dil,lin 200
run 1 4 star_h1_R4_nt300 star 300
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
