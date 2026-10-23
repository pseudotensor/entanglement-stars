#!/bin/bash
# the sigma/a 4 pocket interval (R2 not the negative set on 12.75 to 14.5): slices 13 and 14; queued after run_ls_v3.sh.
set -u
PY=${QI2_ROOT}/python
D=${QI2_ROOT}/localsea
cd ${QI2_ROOT}/localsea
while [ ! -e run_ls_v3.done ]; do sleep 60; done
R() { nice -n 10 $PY "$@"; }
CO4=0.003048063693947,-0.01110837131864,-0.001773874623294
S4=${QI2_ROOT}/orbitals_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0
for T in 13 14 12; do
  R $D/ls_chain_audit_v1.py --mode slice --slice ${S4}_T00${T}.00.npz --coef $CO4 --out audit --tag s4_T$T > logs/audit_s4_T$T.log 2>&1
done
echo DONE > run_ls_v4.done
