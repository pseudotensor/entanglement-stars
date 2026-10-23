#!/bin/bash
# S6 (J4) registry diagnostic: obs_ball_cut_window_v1.py, the exact unit-window average of S over every distinct ball in [R - 1/2, R + 1/2]
# against the eight-midpoint registry mean, at R 12, 16, 20, 24, box 6R, SVD route; 2 BLAS threads (the production holds 12), nice 10.
# Usage on the box: cd ${QI2_ROOT}/J4BC; setsid nohup bash launch_obs_ball_cut_window_v1.sh > logs/window_launcher_v1.log 2>&1 < /dev/null &
cd ${QI2_ROOT}/J4BC
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "window start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> runs/status_window_v1.txt
nice -n 10 ${QI2_ROOT}/python -I ${QI2_ROOT}/obs_ball_cut_window_v1.py --R 12 16 20 24 --box-ratio 6 \
  --solver svd --out-jsonl runs/window_v1.jsonl --md runs/window_v1.md > logs/window_v1.log 2>&1 < /dev/null
echo "window exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> runs/status_window_v1.txt
