#!/bin/bash
# GPU 0: the (0.1, 1/4) family on engine_star_v5 (version 4's family with the seed and scan fixes), ell 8, the fraction grid with bisection,
# the ell-control 4, 8, 16, the closed counts (SPEC_final4b_engine.md secs. 12.1, 13.5).  One GPU, nice 10, 1 thread.
GPU=${1:-0}
cd ${QI2_ROOT}/engine || exit 1
mkdir -p Sfam5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "S-fam5 (0.1, 1/4) ell 8 start $(date -u +%H:%M:%S) UTC (gpu $GPU)" >> status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_star_v5.py --m 0.1 --g 0.25 --ell 8 --nsites 1536 --eos EOS/eos_v4_tables.npz \
  --ovpg OVPG/ovpg_families.json --ell-control 4,8,16 --closed-count --gpu $GPU --out Sfam5 --tag m0.1_g0.25_ell8 > Sfam5/m0.1_g0.25_ell8.out 2>&1
echo "S-fam5 (0.1, 1/4) exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
