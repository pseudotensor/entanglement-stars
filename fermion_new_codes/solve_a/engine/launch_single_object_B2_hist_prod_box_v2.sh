#!/bin/bash
# launch_single_object_B2_hist_prod_box_v2.sh (builder analyst, the HORIZON version 2 builder): the production's history run of
# launch_single_object_B2_hist_dust_box_v1.sh again (its first call, B2_hist_prod, ran out of GPU memory at 02:03:33 UTC at the end-of-history
# eigendecomposition of T/r_s 30.0 kappa 1, beside the S11 gates and a slices job on the same A6000; no npz was written): engine_single_object_B_v2.py
# --mode history on box GPU 0, the production's records to T/r_s 40 and #36 B's continuation's after 40 from the snapshot S4one/v2/inputs/ext_snap1/
# (copied md5-checked at 01:58:20 UTC, records to T/r_s 47.5), T/r_s 15.0 and 30.0, kappa 1 and 2, the 18 columns, the cubic control on the first
# 8 r_s, the signed readout every 1 r_s; started when the B2 chain (the sigma/a 8 history and both dust runs) and the S11 gates have ended, so
# that it has the GPU's memory beside the S11 segment only. Outputs S4one/v2/B2_hist_prod_b.json and .npz (new names; the failed call's json is kept).
# Usage (from ${QI2_ROOT}/engine): nohup bash code/launch_single_object_B2_hist_prod_box_v2.sh &
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PAGE_V6_PATH=${QI2_ROOT}/v6
PY=${QI2_ROOT}/python
S=S4one/v2/status.txt
until grep -q "B2 hist dust all done" $S 2>/dev/null && grep -q "gates done" S11cert/status.txt 2>/dev/null; do sleep 30; done
D=S4one/v2/inputs/ext_snap1
echo "B2_hist_prod_b start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_single_object_B_v2.py --mode history --gpu 0 --record H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json \
    --fields H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-module engine_fields_v14 \
    --record2 $D/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json --fields2 $D/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz --switch-T-rs 40 \
    --launch-T 15.0,30.0 --kappas 1,2 --T-end-rs 100 --cubic-window-rs 8 --signed-every-rs 1 --out-json S4one/v2/B2_hist_prod_b.json --out-npz S4one/v2/B2_hist_prod_b.npz \
    > S4one/v2/log_B2_hist_prod_b.log 2>&1
echo "B2_hist_prod_b exit $? $(date -u +%H:%M:%S) UTC" >> $S
