#!/bin/bash
# launch_single_object_B5_prep_box_v1.sh (builder analyst; the revised S4's B.5 columns): engine_single_object_B5_prep_v1.py on the box CPU
# (nice 10, 8 threads): the production's stored slices at T/r_s 12.0 (advanced one record to 12.125 by B's propagation) and 15.0, kappa 1
# and 2; output S4one/B5_orbitals_v1.npz (the orbitals' columns, for engine_single_object_B_v1.py --mode orbitals on pod 2) and its json.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PAGE_V6_PATH=${QI2_ROOT}/v6
D=H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11
echo "B5prep start $(date -u +%H:%M:%S) UTC" >> S4one/status.txt
nice -n 10 $PY code/engine_single_object_B5_prep_v1.py --record H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json \
    --fields H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --fields-module engine_fields_v14 \
    --slices $D/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0012.00.npz,$D/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0015.00.npz \
    --targets 12.125,15.0 --tags T12p125,T15p000 --kappas 1,2 --gpu -1 --out-npz S4one/B5_orbitals_v1.npz --out-json S4one/B5_orbitals_prep_v1.json \
    > S4one/B5_orbitals_prep_v1.log 2>&1
echo "B5prep exit $? $(date -u +%H:%M:%S) UTC" >> S4one/status.txt
