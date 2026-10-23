#!/bin/bash
# S6 (J4) dense controls, second launcher: the lane's strand-3 dense code (balls.py measure --method dense, unmodified, CPU) at the
# review's ball convention for its integer-radius controls.  obs_ball_cut_v1.py validate showed the review's R 4 and R 6 values are those of
# the strict ball |x| < R (166.3552145607 at |x| < 4, 401.1470249839 at |x| < 6, against the review's 166.35521456 and 401.14702498), while
# balls.py includes |x| = R; so balls.py runs here with --radius 3.99 (the ball |x|^2 <= 15) and --radius 5.99 (|x|^2 <= 35).  Projected
# from ctrl/ctrl_time_v1.json: about 3 min and 3.6 h (one column 0.18 s and 3.56 s).  2 BLAS threads each, nice 10.
# Usage on the box: cd ${QI2_ROOT}/J4BC; setsid nohup bash launch_obs_ball_cut_ctrl_v2.sh > ctrl/launcher_v2.log 2>&1 < /dev/null &
cd ${QI2_ROOT}/J4BC
PY=${QI2_ROOT}/python
ORIENT=${QI2_ROOT}/orient
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run() {
  echo "ctrl $3 start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> ctrl/status_ctrl_v1.txt
  nice -n 10 $PY $ORIENT/balls.py measure --radius $1 --box $2 --method dense --mass 0 --center 0 0 0 --gpu -1 \
    --output ctrl/balls_dense_$3.json > ctrl/balls_dense_$3.log 2>&1 < /dev/null
  echo "ctrl $3 exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> ctrl/status_ctrl_v1.txt
}
run 3.99 64 R3.99_L64 &
run 5.99 128 R5.99_L128 &
wait
echo "ctrl v2 all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> ctrl/status_ctrl_v1.txt
