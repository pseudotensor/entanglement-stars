#!/bin/bash
# items 3/4 stage 2, second chain: d = 2 in long-double precise mode (PM1's primary spacing), then the d = 3 clock-only finite-amplitude
# supplement (each direction at its own subspace eta_max). Box CPU, 8 threads, nice 10, one at a time.
set -u
PY=${QI2_ROOT}/python
S=${QI2_ROOT}/s2_items34_v1.py
cd ${QI2_ROOT}/items34
nice -n 10 $PY -u $S --d 2 --precise --finite 5 > s2_items34_d2.log 2>&1
nice -n 10 $PY -u $S --d 3 --finite 3 --dirs worst_N --tag-suffix _clock > s2_items34_d3_clock.log 2>&1
echo DONE > s2_chain_v2.done
