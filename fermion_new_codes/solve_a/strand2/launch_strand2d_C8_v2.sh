#!/bin/bash
# Strand 2d on C8, second run (strand2d_jet_v2.py: levels stored per point): the vacuum jet (H = 1) and the matter jets at (g_M, g_B) = (0, 0) and (0.2, 1), Lambda 1, m 0.1,
# q in {0, pi}, h in {1e-3, 1e-4}; CPU (the (2,0) block is 120-dimensional), 2 threads.
set -u
D=${QI2_ROOT}/s2d_jet
cd $D || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p C8run2
date -u '+start %Y-%m-%d %H:%M:%S UTC' > C8run2/status.txt
nice -n 10 ${QI2_ROOT}/python code/strand2d_jet_v2.py --out C8run2 --lattice C8 --Hs 1 --couplings "0,0;0.2,1" --device cpu > C8run2/jet.log 2>&1
echo "jet exit $?" >> C8run2/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> C8run2/status.txt
echo DONE >> C8run2/status.txt
