#!/bin/bash
# items 3/4 stage 2, final chain (HS residual operator-error accounting): d = 4 and d = 3 bounds (double Gram, certified), d = 2 in the
# long-double precise mode with finite-amplitude evaluations, then the d = 3 clock-only finite supplement. The d = 3 full-direction
# finite-amplitude rows are from the first run (old/s2_items34_d3_run1.json; floating evaluations at stated absolute eta).
set -u
PY=${QI2_ROOT}/python
S=${QI2_ROOT}/s2_items34_v1.py
cd ${QI2_ROOT}/items34
nice -n 10 $PY -u $S --d 4 --finite 0 > s2_items34_d4.log 2>&1
nice -n 10 $PY -u $S --d 3 --finite 0 > s2_items34_d3.log 2>&1
nice -n 10 $PY -u $S --d 2 --precise --finite 5 > s2_items34_d2.log 2>&1
nice -n 10 $PY -u $S --d 3 --finite 3 --dirs worst_N --tag-suffix _clock > s2_items34_d3_clock.log 2>&1
nice -n 10 $PY -u ${QI2_ROOT}/s4_read_v1.py > s4_read_v1.log 2>&1
echo DONE > s2_chain_v3.done
