#!/bin/bash
# reordered remainder of run_ls_v2..v5 (inbox #447): pocket slices and the real-time check first. Box CPU, 8 threads, nice 10.
set -u
PY=${QI2_ROOT}/python
D=${QI2_ROOT}/localsea
cd ${QI2_ROOT}/localsea
R() { nice -n 10 $PY "$@"; }
CO4=0.003048063693947,-0.01110837131864,-0.001773874623294
S4=${QI2_ROOT}/orbitals_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0
for T in 13 14; do
  R $D/ls_chain_audit_v1.py --mode slice --slice ${S4}_T00${T}.00.npz --coef $CO4 --out audit --tag s4_T$T > logs/audit_s4_T$T.log 2>&1
done
R $D/ls_exact_v1.py --mode evolve --Lx 16 --Ly 1 --ny 8 --nz 8 --B 1e-4 --tau 30 --T 150 --dt 0.05 --out exact --tag evolve_L16 > logs/exact_evolve_L16.log 2>&1
for L in 1 2; do
  R $D/ls_chain_audit_v1.py --mode scale --slice ${S4}_T0015.00.npz --lam $L --out audit --tag s4_T15_lam$L > logs/audit_s4_T15_lam$L.log 2>&1
done
RD8=${QI2_ROOT}/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40
R $D/ls_chain_audit_v1.py --mode rates --json $RD8.json --fields ${RD8}_fields.npz --handoff ${RD8}_handoff.npz --T0 13.0 --T1 16.0 --out audit --tag rates_s8 > logs/audit_rates_s8.log 2>&1
R $D/ls_chain_audit_v1.py --mode scale --slice ${S4}_T0015.00.npz --lam 3 --out audit --tag s4_T15_lam3 > logs/audit_s4_T15_lam3.log 2>&1
R $D/ls_exact_v1.py --mode rate2 --Lx 96 --Ly 1 --ny 40 --nz 40 --j 1 --A 1e-3 --B 1e-3 --dt 1e-2 --out exact --tag rate2_L96_j1 > logs/exact_rate2_L96_j1.log 2>&1
echo DONE > run_ls_v6.done
