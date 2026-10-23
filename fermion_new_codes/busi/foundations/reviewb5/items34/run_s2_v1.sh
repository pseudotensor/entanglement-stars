#!/bin/bash
# items 3/4 stage 2 chain, box CPU, 8 threads, nice 10, one at a time.
set -u
PY=${QI2_ROOT}/python
S=${QI2_ROOT}/s2_items34_v1.py
cd ${QI2_ROOT}/items34
nice -n 10 $PY -u $S --d 4 --finite 0 > s2_items34_d4.log 2>&1
nice -n 10 $PY -u $S --d 3 --finite 5 > s2_items34_d3.log 2>&1
nice -n 10 $PY -u $S --d 2 --finite 5 > s2_items34_d2.log 2>&1
echo DONE > s2_chain_v1.done
