#!/bin/bash
# a third q for the nonlinear rate terms (q = 2 pi/96), queued after run_ls_v4.sh; box CPU, 8 threads, nice 10.
set -u
PY=${QI2_ROOT}/python
D=${QI2_ROOT}/localsea
cd ${QI2_ROOT}/localsea
while [ ! -e run_ls_v4.done ]; do sleep 60; done
nice -n 10 $PY $D/ls_exact_v1.py --mode rate2 --Lx 96 --Ly 1 --ny 40 --nz 40 --j 1 --A 1e-3 --B 1e-3 --dt 1e-2 --out exact --tag rate2_L96_j1 > logs/exact_rate2_L96_j1.log 2>&1
nice -n 10 $PY $D/ls_exact_v1.py --mode rate2 --Lx 32 --Ly 1 --ny 56 --nz 56 --j 1 --A 1e-3 --B 1e-3 --dt 1e-2 --out exact --tag rate2_L32_j1_n56 > logs/exact_rate2_L32_j1_n56.log 2>&1
echo DONE > run_ls_v5.done
