#!/bin/bash
# RUN_ORDER_S10_v2 step 8 on the pod (builder): item e for every member, operator route, closed multiplets (odd members)
# first, at most 4 processes at a time with 4 threads each; the solve lane's records read from read-only copies.
cd ${QI2_ROOT}/hf
OUT=${QI2_ROOT}/hf; mkdir -p $OUT/star_var_v2_e
STAR=${QI2_ROOT}/star_m0.1_g0.25_ell8.json; ENG=${QI2_ROOT}/code; MF=${QI2_ROOT}/MF
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
pids=()
for m in 1 3 5 7 9 11 13 15 17 0 2 4 6 8 10 12 14 16; do
  while [ $(for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && echo $p; done | wc -l) -ge 4 ]; do sleep 30; done
  python hf_star_variance_v2.py --star $STAR --member $m --response-mode operator --engine-dir $ENG --mf-dir $MF --threads 4 --out $OUT/star_var_v2_e > $OUT/star_var_v2_e_m$m.log 2>&1 &
  pids+=($!); echo "member $m pid $! $(date -u +%H:%M:%S)" >> $OUT/step8_status.txt
done
wait; echo "all done $(date -u +%H:%M:%S)" >> $OUT/step8_status.txt
