#!/bin/bash
# Strand 5 J5 pilot, run 2 (strand5_w0radial_v4.py; spec 10(p), 10(p'), 10(p'')): the (0.1, 0.5) kernel g = 0 family's member z_c = 0.823379
# (r_s = 80.4), h = 1, R_max = 4 r_s, every j to the resolution limit, cut 4; GPU 2: the flat control, the star series and the dilation
# family; GPU 1: the Gaussians and the lattice forms. One compute process per GPU, nice 10, 2 threads each. Builder analyst.
cd ${QI2_ROOT}/s5_star || exit 1
O=J5pilot2
mkdir -p $O
echo "start $(date -u '+%Y-%m-%d %H:%M:%S') UTC" > $O/status.txt
F=J4run2/lstar_m0.1_aH0.5_kernel_g0_0.json
P=J4run2/lstar_m0.1_aH0.5_kernel_g0_0_profiles.npz
K=${QI2_ROOT}/kernel_L256.json
PY=${QI2_ROOT}/python
( CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/strand5_w0radial_v4.py --family $F --profiles $P --zc 0.823379 --kernel $K --h 1 --rmax-over-rs 4 --what flat,star,dil --eps spec,big --widths 20,60 --gpu 0 --out $O --tag star_h1_R4 > $O/star_h1_R4.log 2>&1; echo "star_h1_R4 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt ) &
( CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/strand5_w0radial_v4.py --family $F --profiles $P --zc 0.823379 --kernel $K --h 1 --rmax-over-rs 4 --what gauss --eps spec,big --widths 20,60 --gpu 0 --out $O --tag gauss_h1 > $O/gauss_h1.log 2>&1; echo "gauss_h1 exit $? $(date -u +%H:%M:%S)" >> $O/status.txt ) &
wait
echo "end $(date -u '+%Y-%m-%d %H:%M:%S') UTC" >> $O/status.txt
echo DONE >> $O/status.txt
