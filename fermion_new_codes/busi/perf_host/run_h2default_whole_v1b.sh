#!/bin/bash
# run_h2default_whole_v1b.sh (busi, analyst; inbox #343): the v36r pair again with h2default_whole_v1b.py (NEWTON without
# --nk-host-release, which v36r lacks: v1 exited 2 on it), on run_h2default_whole_v1.sh's code dirs, into whole/runs_b. Box CPU, nice 10, 2 x 4 threads.
set -u
B=${QI2_ROOT}/H2default; W=$B/whole; PY=${QI2_ROOT}/python
$PY -I $B/code/h2default_whole_v1b.py --code16 $W/code16 --code16c $W/code16c --out $W/runs_b --selftest ${QI2_ROOT}/engine_collapse_v36z_selftest.py --threads 4 --only RN
echo done > $W/WHOLE_B_DONE
