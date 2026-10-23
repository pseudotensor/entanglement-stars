#!/bin/bash
# after run_ls_v6.sh (inbox #449): the lambda = 3 scaling run from the lambda = 1 output's fields (the slice file left the box at
# 14:00 UTC 22 Oct, so v6's lambda = 3 step cannot read it), with a lambda = 1 recomputation first as the check. Box CPU, 8 threads, nice 10.
set -u
PY=${QI2_ROOT}/python
D=${QI2_ROOT}/localsea
cd ${QI2_ROOT}/localsea
while [ ! -e run_ls_v6.done ]; do sleep 60; done
F=audit/ls_chain_audit_v1_s4_T15_lam1.npz
nice -n 10 $PY -B $D/ls_chain_scale_v2.py --fields_npz $F --lam 1 --out audit --tag s4_T15_lam1_fromnpz > logs/audit_s4_T15_lam1_fromnpz.log 2>&1
nice -n 10 $PY -B $D/ls_chain_scale_v2.py --fields_npz $F --lam 3 --out audit --tag s4_T15_lam3_fromnpz > logs/audit_s4_T15_lam3_fromnpz.log 2>&1
echo DONE > run_ls_v7.done
