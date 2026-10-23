#!/bin/bash
# busi #469 (leakage option b): operators for prod 35 and s8 14 (s46v31_mass2save_v1.py, no R_out), one after another on box GPU 1 (cuda:1).
# Measured: about 1565 s per slice on cuda:1 (s46v3 prod 35 log).
PY=${QI2_ROOT}/python
B=${QI2_ROOT}/s46v31
cd $B/code
for s in "prod 35" "s8 14"; do
  set -- $s
  echo "$(date -u +%FT%TZ) start $1 $2" >> $B/logs/mass2save_q2.status
  nice -n 10 $PY -B s46v31_mass2save_v1.py --slice $1 $2 --device cuda:1 --threads 8 > $B/logs/mass2save_${1}_T$2.log 2>&1
  echo "$(date -u +%FT%TZ) end rc=$? $1 $2" >> $B/logs/mass2save_q2.status
done
echo "$(date -u +%FT%TZ) queue done" >> $B/logs/mass2save_q2.status
