#!/bin/bash
# engine_massfluct_v3.py on the (0.1, 1/4) family at ell 8 (Sfam5/star_m0.1_g0.25_ell8.json), members 1, 3, 5, 7, 9 in parallel (2 threads
# each, nice 10, CPU), then 11, 13, 15, 17 in parallel once version 2's chain has finished those members (its records are the comparison).
# Outputs MF/massfluct3_sfam5_m<k>.json and MF/v3_sfam5_m<k>.log. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
run() { OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/engine_massfluct_v3.py --star Sfam5/star_m0.1_g0.25_ell8.json --member $1 --out MF --tag sfam5_m$1 > MF/v3_sfam5_m$1.log 2>&1 < /dev/null; echo "massfluct3 m$1 exit $? $(date -u +%H:%M:%S) UTC" >> MF/status_v3.txt; }
echo "massfluct3 batch 1 start $(date -u +%H:%M:%S) UTC" >> MF/status_v3.txt
for k in 1 3 5 7 9; do run $k & done
wait
until [ -f MF/massfluct2_sfam5_m17.json ]; do sleep 120; done
echo "massfluct3 batch 2 start $(date -u +%H:%M:%S) UTC" >> MF/status_v3.txt
for k in 11 13 15 17; do run $k & done
wait
echo "massfluct3 DONE $(date -u +%H:%M:%S) UTC" >> MF/status_v3.txt
