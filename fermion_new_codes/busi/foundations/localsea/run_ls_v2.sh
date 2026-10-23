#!/bin/bash
# local-sea audit (inbox #447), queued after run_ls_v1.sh: box CPU only, one job at a time, 8 threads, nice 10.
set -u
PY=${QI2_ROOT}/python
D=${QI2_ROOT}/localsea
cd ${QI2_ROOT}/localsea
mkdir -p zone audit logs
while [ ! -e run_ls_v1.done ]; do sleep 60; done
R() { nice -n 10 $PY "$@"; }
CO4=0.003048063693947,-0.01110837131864,-0.001773874623294
SL=${QI2_ROOT}/slices
S4=$SL/orbitals_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0
S8=$SL/orbitals_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0
# the named slice records first (reviewer B: 15, 16 at sigma/a 4; 14, 16 at sigma/a 8) and one pre-gate record (7 at sigma/a 4)
R $D/ls_chain_audit_v1.py --mode slice --slice ${S4}_T0015.00.npz --coef $CO4 --out audit --tag s4_T15 > logs/audit_s4_T15.log 2>&1
R $D/ls_chain_audit_v1.py --mode slice --slice ${S4}_T0016.00.npz --coef $CO4 --out audit --tag s4_T16 > logs/audit_s4_T16.log 2>&1
R $D/ls_chain_audit_v1.py --mode slice --slice ${S8}_T0014.00.npz --coef $CO4 --out audit --tag s8_T14 > logs/audit_s8_T14.log 2>&1
R $D/ls_chain_audit_v1.py --mode slice --slice ${S8}_T0016.00.npz --coef $CO4 --out audit --tag s8_T16 > logs/audit_s8_T16.log 2>&1
R $D/ls_chain_audit_v1.py --mode slice --slice ${S4}_T0007.00.npz --coef $CO4 --out audit --tag s4_T07 > logs/audit_s4_T07.log 2>&1
# fine K lists for the linear rate response (the a^-2 coefficient c3 and the K^3 / K^3 ln K order)
R $D/ls_zone_v1.py --m 0.2 --n-list 192 --parts A,D --k-list 0.0025,0.005,0.01,0.02,0.04,0.08,0.16 --out zone --tag m0p2_Dfine > logs/zone_m0p2_Dfine.log 2>&1
R $D/ls_zone_v1.py --m 0.0 --n-list 384 --parts A,D --k-list 0.0025,0.005,0.01,0.02,0.04,0.08,0.16 --out zone --tag m0_Dfine > logs/zone_m0_Dfine.log 2>&1
# the reference's rate at the outer root, every record of the gate window, from the stored applied fields
RD4=${QI2_ROOT}/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40
RD8=${QI2_ROOT}/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40
R $D/ls_chain_audit_v1.py --mode rates --json $RD4.json --fields ${RD4}_fields.npz --handoff ${RD4}_handoff.npz --T0 12.0 --T1 17.5 --out audit --tag rates_s4 > logs/audit_rates_s4.log 2>&1
R $D/ls_chain_audit_v1.py --mode rates --json $RD8.json --fields ${RD8}_fields.npz --handoff ${RD8}_handoff.npz --T0 12.625 --T1 16.5 --out audit --tag rates_s8 > logs/audit_rates_s8.log 2>&1
# the scaling test on the sigma/a 4 record 15: lambda = 1, 2, 3 (the chain sea's second, fourth and higher orders)
for L in 1 2 3; do
  R $D/ls_chain_audit_v1.py --mode scale --slice ${S4}_T0015.00.npz --lam $L --out audit --tag s4_T15_lam$L > logs/audit_s4_T15_lam$L.log 2>&1
done
echo DONE > run_ls_v2.done
