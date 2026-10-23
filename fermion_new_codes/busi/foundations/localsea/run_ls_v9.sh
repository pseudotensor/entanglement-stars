#!/bin/bash
# inbox #455 (ReviewB_found7_v1 sec. 1 D: a further smaller q): after run_ls_v8.sh, the fourth wavevector q = 2 pi/128 (j = 1,
# 40 transverse points, as L = 32, 64, 96), then the four-point reader. Box CPU, 8 threads, nice 10.
set -u
PY=${QI2_ROOT}/python
D=${QI2_ROOT}/localsea
cd ${QI2_ROOT}/localsea
while [ ! -e run_ls_v8.done ]; do sleep 60; done
nice -n 10 $PY -B $D/ls_exact_v1.py --mode rate2 --Lx 128 --Ly 1 --ny 40 --nz 40 --j 1 --A 1e-3 --B 1e-3 --dt 1e-2 --out exact --tag rate2_L128_j1 > logs/exact_rate2_L128_j1_v9.log 2>&1
nice -n 10 $PY -B $D/ls_rate3q_read_v2.py > logs/rate3q_read_v2.log 2>&1
echo DONE > run_ls_v9.done
