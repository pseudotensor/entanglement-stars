#!/bin/bash
# busi #469 (leakage option b, box only): Lanczos B_max on prod 15 (now), prod 35 and s8 14 (each after its operators are saved).
# Sectors: single16 (the dense check), single18, single20, copy2 (kappa 1 x 2 copies, 10 + 10), chan2 (kappa 1 + kappa 2, 10 + 10).
# GPU 1 (cuda:1) only for the 4096-dimensional one-body steps; the sparse work runs on the CPU (8 threads, nice 10).
PY=${QI2_ROOT}/python
B=${QI2_ROOT}/s46v31
mkdir -p $B/lanczos
cd $B/code
run() { echo "$(date -u +%FT%TZ) start $1 $2" >> $B/logs/lanczos.status
  nice -n 10 $PY -B s46v31_leak_lanczos_v1.py --slice $1 $2 --ops $B/ops/$3 --device cuda:1 --threads 8 > $B/logs/lanczos_${1}_T$2.log 2>&1
  echo "$(date -u +%FT%TZ) end rc=$? $1 $2" >> $B/logs/lanczos.status; }
run prod 15 prod_T0015.00_ops.npz
until grep -q "end rc=.* prod 35" $B/logs/mass2save_q2.status 2>/dev/null; do sleep 30; done
grep -q "end rc=0 prod 35" $B/logs/mass2save_q2.status && run prod 35 prod_T0035.00_ops.npz
until grep -q "end rc=.* s8 14" $B/logs/mass2save_q2.status 2>/dev/null; do sleep 30; done
grep -q "end rc=0 s8 14" $B/logs/mass2save_q2.status && run s8 14 s8_T0014.00_ops.npz
echo "$(date -u +%FT%TZ) queue done" >> $B/logs/lanczos.status
