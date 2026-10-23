#!/bin/bash
# exact real-time evolution check of the adiabatic rate formula (inbox #447), queued after run_ls_v2.sh; box CPU, 8 threads, nice 10.
set -u
PY=${QI2_ROOT}/python
D=${QI2_ROOT}/localsea
cd ${QI2_ROOT}/localsea
while [ ! -e run_ls_v2.done ]; do sleep 60; done
R() { nice -n 10 $PY "$@"; }
R $D/ls_exact_v1.py --mode evolve --Lx 16 --Ly 1 --ny 8 --nz 8 --B 1e-4 --tau 30 --T 150 --dt 0.05 --out exact --tag evolve_L16 > logs/exact_evolve_L16.log 2>&1
R $D/ls_exact_v1.py --mode evolve --Lx 24 --Ly 1 --ny 8 --nz 8 --B 1e-4 --tau 30 --T 150 --dt 0.05 --out exact --tag evolve_L24 > logs/exact_evolve_L24.log 2>&1
echo DONE > run_ls_v3.done
