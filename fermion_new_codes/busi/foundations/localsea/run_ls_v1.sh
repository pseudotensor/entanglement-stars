#!/bin/bash
# local-sea coefficients (inbox #447): box CPU only, one job at a time, 8 threads, nice 10. Outputs in ${QI2_ROOT}/.
set -u
PY=${QI2_ROOT}/python
D=${QI2_ROOT}/localsea
cd ${QI2_ROOT}/localsea
mkdir -p zone exact logs
R() { nice -n 10 $PY "$@"; }
# exact-diagonalization second route (finite lattices, same momentum grids)
for md in two three rate1; do
  R $D/ls_exact_v1.py --mode $md --Lx 12 --Ly 12 --nz 8 --out exact --tag ${md}_L12 > logs/exact_${md}_L12.log 2>&1
  R $D/ls_exact_v1.py --mode $md --Lx 16 --Ly 16 --nz 6 --out exact --tag ${md}_L16 > logs/exact_${md}_L16.log 2>&1
done
# zone integrals, heavy field ma = 0.2
R $D/ls_zone_v1.py --m 0.2 --n-list 48,64,96,128,192 --parts A,B,D --t-list 0.02,0.04,0.08 --out zone --tag m0p2_ABD > logs/zone_m0p2_ABD.log 2>&1
for n3 in 48 64 96; do
  R $D/ls_zone_v1.py --m 0.2 --n-list 128 --parts A,C --n3 $n3 --nconf 24 --t3-list 0.005,0.01,0.02 --out zone --tag m0p2_C_n$n3 > logs/zone_m0p2_C_n$n3.log 2>&1
done
# light field m = 0 (substitution grid; t -> 0 Richardson)
R $D/ls_zone_v1.py --m 0.0 --n-list 64,96,128,192,256,384 --parts A,B,D --t-list 0.01,0.02,0.04 --out zone --tag m0_ABD > logs/zone_m0_ABD.log 2>&1
for n3 in 64 96 128; do
  R $D/ls_zone_v1.py --m 0.0 --n-list 192 --parts A,C --n3 $n3 --nconf 24 --t3-list 0.0025,0.005,0.01 --out zone --tag m0_C_n$n3 > logs/zone_m0_C_n$n3.log 2>&1
done
# intermediate masses (the m -> 0 cross-check)
for mm in 0.1 0.05 0.025; do
  R $D/ls_zone_v1.py --m $mm --n-list 128,192,256 --parts A,C,D --n3 96 --nconf 24 --t3-list 0.0025,0.005,0.01 --out zone --tag m${mm}_ACD > logs/zone_m${mm}_ACD.log 2>&1
done
echo DONE_A > run_ls_v1.stageA
# nonlinear rate terms, exact adiabatic on 1D modulations, q -> 0 and size checks
for LJ in "32 1" "64 2" "64 1" "128 2" "128 1"; do
  set -- $LJ
  R $D/ls_exact_v1.py --mode rate2 --Lx $1 --Ly 1 --ny 40 --nz 40 --j $2 --A 1e-3 --B 1e-3 --dt 1e-2 --out exact --tag rate2_L$1_j$2 > logs/exact_rate2_L$1_j$2.log 2>&1
done
R $D/ls_exact_v1.py --mode rate2 --Lx 64 --Ly 1 --ny 24 --nz 24 --j 1 --A 1e-3 --B 1e-3 --dt 1e-2 --out exact --tag rate2_L64_j1_n24 > logs/exact_rate2_L64_j1_n24.log 2>&1
echo DONE > run_ls_v1.done
