#!/bin/bash
# busi #467 item 2: the leakage pilot on prod 15, n = 12, 14, 16 (half filling), box GPU 1 (cuda:1), after the operators are saved.
PY=${QI2_ROOT}/python
B=${QI2_ROOT}/s46v31
until grep -q "end rc=" $B/logs/mass2save_p15.status 2>/dev/null; do sleep 30; done
grep -q "end rc=0" $B/logs/mass2save_p15.status || { echo "$(date -u +%FT%TZ) operators run failed; pilot not started" >> $B/logs/leak_p15.status; exit 1; }
cd $B/code
echo "$(date -u +%FT%TZ) start" >> $B/logs/leak_p15.status
nice -n 10 $PY -B s46v31_leak_pilot_v1.py --slice prod 15 --ops $B/ops/prod_T0015.00_ops.npz --n 12,14,16 --device cuda:1 --threads 8 > $B/logs/leak_p15.log 2>&1
echo "$(date -u +%FT%TZ) end rc=$?" >> $B/logs/leak_p15.status
