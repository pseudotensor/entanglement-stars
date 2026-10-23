#!/bin/bash
# Engine job S-fam, the (m a, g) = (0.1, 1/4) family at ell = 8 on engine_star_v3 (zero-temperature filling at fixed N_B, the last
# degenerate set fractional; Anderson mixing; the continuum start at E_F/m 0.95), N_B from 200 to the stable branch's end, the ell-control
# 4, 8, 16 at the maximum member and at the member nearest E_F/m 0.95.  PROVISIONAL on the lead's filling ruling (turn 19, pending):
# the builder runs the red team's "smallest clean correction" (zero-temperature filling), labelled.  One GPU, nice 10, 1 thread.
GPU=${1:-0}
cd ${QI2_ROOT}/engine || exit 1
mkdir -p Sfam
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "S-fam (0.1, 1/4) ell 8 zeroT_NB start $(date -u +%H:%M:%S) UTC (gpu $GPU; provisional on the filling ruling)" >> status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_star_v3.py --m 0.1 --g 0.25 --ell 8 --filling zeroT_NB \
  --NB-list 200,300,400,500,600,700,800,900,950,1000,1030,1050,1070,1080,1090 --EF-start 0.95 --EF-stop 0.95 \
  --start-eos EOS/eos_v4_tables.npz --nsites 1536 --ell-control 4,8,16 --gpu $GPU --out Sfam --tag m0.1_g0.25_ell8_zT > Sfam/m0.1_g0.25_ell8_zT.out 2>&1
echo "S-fam (0.1, 1/4) exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
