#!/bin/bash
# S6 (J4) dense controls with the lane's strand-3 dense code (balls.py measure --method dense of the paper's orient package, unmodified,
# md5 9c13956b9279935b3ac3d1b634b78d73, on the CPU: the box GPUs are busy).  Site-centred, antiperiodic, m 0:
#   R 4 L 64 (review 166.35521456), R 6.5 L 80 (review 497.4638266), R 6 L 128 (review 401.14702498).
# Timed by obs_ball_cut_ctrl_time_v1.py (ctrl/ctrl_time_v1.json): one dense column 0.18, 0.40, 3.56 s, so the dense builds take about
# 3 min, 31 min and 3.7 h (single-threaded FFT matvecs), plus an eigvalsh at D = 1028, 4756, 3700.  Three processes, 2 BLAS threads each,
# nice 10.  Usage on the box: cd ${QI2_ROOT}/J4BC; setsid nohup bash launch_obs_ball_cut_ctrl_v1.sh > ctrl/launcher_v1.log 2>&1 < /dev/null &
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
run 4 64 R4_L64 &
run 6.5 80 R6.5_L80 &
run 6 128 R6_L128 &
wait
echo "ctrl all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> ctrl/status_ctrl_v1.txt
