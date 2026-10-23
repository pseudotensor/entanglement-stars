#!/bin/bash
# inbox #468 (ReviewA_found8_v1 sec. 3, ReviewB_found8_v1 sec. 3): the smallest-q size control, q = 2 pi/128 on L = 256 with j = 2
# (against L = 128, j = 1), A = B = 1e-3, 40 transverse points, then the size-control reader. Box CPU, 8 threads, nice 10.
set -u
PY=${QI2_ROOT}/python
D=${QI2_ROOT}/localsea
cd ${QI2_ROOT}/localsea
date -u +"start %FT%TZ" > logs/run_ls_v10.times
nice -n 10 $PY -B $D/ls_exact_v1.py --mode rate2 --Lx 256 --Ly 1 --ny 40 --nz 40 --j 2 --A 1e-3 --B 1e-3 --dt 1e-2 --out exact --tag rate2_L256_j2 > logs/exact_rate2_L256_j2_v10.log 2>&1
date -u +"rate2 end %FT%TZ" >> logs/run_ls_v10.times
nice -n 10 $PY -B $D/ls_sizectl_read_v1.py > logs/sizectl_read_v1.log 2>&1
date -u +"reader end %FT%TZ" >> logs/run_ls_v10.times
echo DONE > run_ls_v10.done
