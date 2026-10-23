#!/bin/bash
# J2 version 2 (SPEC_final4b_observables.md sec. 8.2): the decisive test at L = 2048 and 4096 on the CPU (2 threads; L = 8192 on a GPU when
# one frees), W = 128, 256, 512, the profiles (0, 0.6) and (-0.5, 0.3), both cuts, m_c = 0 and 0.05.
cd ${QI2_ROOT}/obs || exit 1
mkdir -p J2v2
for L in 2048 4096; do
  echo "J2v2 L $L start $(date -u +%H:%M:%S) UTC (CPU, 2 threads)" >> status.txt
  OMP_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/obs_bw_modular_v2.py --L $L --out-json J2v2/J2v2_L$L.json --md J2v2/J2v2_L$L.md > J2v2/J2v2_L$L.log 2>&1
  echo "J2v2 L $L exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
done
