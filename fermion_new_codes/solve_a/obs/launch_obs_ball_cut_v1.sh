#!/bin/bash
# S6 (J4) production on the box CPU (builder analyst): obs_ball_cut_v1.py, validated on the review's three dense controls (val/).
# Radii R + j/16, j in {-7, -5, -3, -1, 1, 3, 5, 7} (the eight registry offsets; at these radii no site has |x| = R + j/16, so |x| <= R and
# |x| < R give the same ball), boxes L = 4R, 6R, 8R, antiperiodic, site-centred, m 0.
#   1. counts: exact multiplicities and block sizes for R in {12, ..., 64} (no kernel; minutes)
#   2. R 12 16 20 24 32 by the SVD route (gesdd), every block also by the Gram route (--check-gram): timed at R 32, L 256, j +7:
#      65 s per (offset, box) with 12 threads, so about 26 min for R 32 and under 10 min for the rest
#   3. R 40 by the Gram route (5 times faster than gesdd at R 32; entropies equal to 5e-11 per block there): about 115 s per (offset,
#      box) projected (blocks to 8900, assembly about 30 s), about 46 min
#   4. R 40, j +7, L 320 by the SVD route (the Gram route's spot check at that size, about 7 min)
#   5. bench: real FP64 gesdd and Gram eigvalsh at m 2000, 4000, 8000, complex128 eigh n 4096 (the lane's A6000/H100 calibration point)
# 12 BLAS threads (the balls.py controls use about 4 cores), nice 10.
# Usage on the box: cd ${QI2_ROOT}/J4BC; setsid nohup bash launch_obs_ball_cut_v1.sh > logs/launcher_v1.log 2>&1 < /dev/null &
cd ${QI2_ROOT}/J4BC
PY=${QI2_ROOT}/python
CODE=${QI2_ROOT}/obs_ball_cut_v1.py
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
st() { echo "$1 $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> runs/status_v1.txt; }
st "counts start"
nice -n 10 $PY -I $CODE counts --R 12 16 20 24 32 40 48 56 62 64 --out-json runs/counts_v1.json > logs/counts_v1.log 2>&1 < /dev/null
st "counts exit $?"
st "svd R12-32 start"
nice -n 10 $PY -I $CODE run --R 12 16 20 24 32 --check-gram --out-dir runs --tag svd_v1 > logs/run_svd_v1.log 2>&1 < /dev/null
st "svd R12-32 exit $?"
st "gram R40 start"
nice -n 10 $PY -I $CODE run --R 40 --solver gram --out-dir runs --tag gram_v1 > logs/run_gram_v1.log 2>&1 < /dev/null
st "gram R40 exit $?"
st "svd R40 check start"
nice -n 10 $PY -I $CODE run --R 40 --offsets16 7 --box-ratios 8 --out-dir runs --tag svd40chk_v1 > logs/run_svd40chk_v1.log 2>&1 < /dev/null
st "svd R40 check exit $?"
st "bench start"
nice -n 10 $PY -I $CODE bench --sizes 2000 4000 8000 --complex-calib --out-json runs/bench_v1.json > logs/bench_v1.log 2>&1 < /dev/null
st "bench exit $?"
st "all done"
