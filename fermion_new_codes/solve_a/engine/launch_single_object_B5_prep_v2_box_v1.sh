#!/bin/bash
# launch_single_object_B5_prep_v2_box_v1.sh (builder analyst, the HORIZON version 2 builder; engine spec 38.4 (1) (ii) and (iv)): the
# columns of B.5 version 2 by engine_single_object_B5_prep_v2.py on box GPU 0 (free until the certified S11 segment; nice 10, 8 CPU threads):
# (1) the production (C3prod19 c19 r11, engine_fields_v14): the T/r_s 12.0 slice advanced one record to 12.125 (the first trapped record) and
#     the T/r_s 15.0 slice, kappa 1 and 2;
# (2) the sigma/a 8 history (C3lad27 c27 ell8, engine_fields_v16): its slices at T/r_s 13.0 (the stored slice nearest after its formation,
#     12.629) and 16.0 (its first episode; it stores no slice at 15), kappa 1 and 2.
# Outputs in S4one/v2/; status lines in S4one/v2/status.txt. Usage (from ${QI2_ROOT}/engine): bash code/launch_single_object_B5_prep_v2_box_v1.sh
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PAGE_V6_PATH=${QI2_ROOT}/v6
PY=${QI2_ROOT}/python
mkdir -p S4one/v2
S=S4one/v2/status.txt
DP=H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11
D8=H100b/states/c27_rs64_A_R2_tau8_ell8_prep4_avg
echo "B5prep_v2_prod start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_single_object_B5_prep_v2.py --record H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json \
    --fields H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-module engine_fields_v14 \
    --slices $DP/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0012.00.npz,$DP/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0015.00.npz \
    --targets 12.125,15.0 --tags T12p125,T15p000 --kappas 1,2 --gpu 0 --out-npz S4one/v2/B5v2_cols_prod.npz --out-json S4one/v2/B5v2_cols_prod.json \
    > S4one/v2/B5v2_cols_prod.log 2>&1
echo "B5prep_v2_prod exit $? $(date -u +%H:%M:%S) UTC" >> $S
echo "B5prep_v2_s8 start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_single_object_B5_prep_v2.py --record H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json \
    --fields H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz --fields-module engine_fields_v16 \
    --slices $D8/orbitals_c27_rs64_A_R2_tau8_ell8_prep4_avg_T0013.00.npz,$D8/orbitals_c27_rs64_A_R2_tau8_ell8_prep4_avg_T0016.00.npz \
    --targets 13.0,16.0 --tags T13p000,T16p000 --kappas 1,2 --gpu 0 --out-npz S4one/v2/B5v2_cols_s8.npz --out-json S4one/v2/B5v2_cols_s8.json \
    > S4one/v2/B5v2_cols_s8.log 2>&1
echo "B5prep_v2_s8 exit $? $(date -u +%H:%M:%S) UTC" >> $S
