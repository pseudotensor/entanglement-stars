#!/bin/bash
# S6 (J4) registry diagnostic, second launcher: obs_ball_cut_window_v1.py at R 28 and 32, box 6R, the Gram route (equal to the SVD
# route to 5e-11 per block at R 32), 2 BLAS threads beside the production's 12 and the controls, nice 10.  Projected about 25 min (R 28,
# about 47 balls) and 60 min (R 32, about 54 balls) at 2 threads.
# Usage on the box: cd ${QI2_ROOT}/J4BC; setsid nohup bash launch_obs_ball_cut_window_v2.sh > logs/window_launcher_v2.log 2>&1 < /dev/null &
cd ${QI2_ROOT}/J4BC
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for R in 28 32; do
  echo "window R$R start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> runs/status_window_v1.txt
  nice -n 10 ${QI2_ROOT}/python -I ${QI2_ROOT}/obs_ball_cut_window_v1.py --R $R --box-ratio 6 \
    --solver gram --out-jsonl runs/window_v2.jsonl --md runs/window_v2.md > logs/window_v2_R$R.log 2>&1 < /dev/null
  echo "window R$R exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> runs/status_window_v1.txt
done
