#!/bin/bash
# RUN_ORDER_S10_v3 step 5 on the pod (builder): the channel control on members 1 and 7 (4 extra sea-only channels).
cd ${QI2_ROOT}/hf
OUT=${QI2_ROOT}/hf; STAR=${QI2_ROOT}/star_m0.1_g0.25_ell8.json; ENG=${QI2_ROOT}/code; MF=${QI2_ROOT}/MF
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for m in 1 7; do
  python hf_star_variance_v3.py --star $STAR --member $m --extra-channels 4 --engine-dir $ENG --mf-dir $MF --threads 8 --out $OUT/star_var_v3_chan > $OUT/star_var_v3_chan_m$m.log 2>&1 &
done
wait; echo "step5 all done $(date -u +%H:%M:%S)" >> $OUT/s10_v3_step5_status.txt
