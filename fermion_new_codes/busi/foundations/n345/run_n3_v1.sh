#!/bin/bash
# N3 runs on the box CPU, one at a time (8 threads), nice 10. Outputs in ${QI2_ROOT}/.
set -u
PY=${QI2_ROOT}/python
S=${QI2_ROOT}/n3_background_v1.py
cd ${QI2_ROOT}/n345
nice -n 10 $PY $S --parent chain --n-parent 128 --K 10 --ell 4 --m 0.2 --g 0.25 --eps 1e-4 --tag K10_n128_eps1e-4 > n3_chain_K10_n128_eps1e-4.log 2>&1
nice -n 10 $PY $S --parent chain --n-parent 128 --K 10 --ell 4 --m 0.2 --g 0.25 --eps 1e-5 --tag K10_n128_eps1e-5 > n3_chain_K10_n128_eps1e-5.log 2>&1
nice -n 10 $PY $S --parent ball --Rb 8 --ell 2 --m 0.2 --g 0.25 --eps 1e-4 --tag Rb8_ell2_eps1e-4 > n3_ball_Rb8_ell2_eps1e-4.log 2>&1
echo DONE > n3_runs_v1.done
