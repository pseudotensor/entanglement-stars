#!/bin/bash
# launch_single_object_A_v1.sh (builder analyst; inbox #41, the revised S4 part A: post-processing of the stored states, no propagation, no
# mutual information): engine_single_object_A_v1.py on the box CPU, nice 10, the two r_s/a 64 families in parallel at 8 threads each (16 in
# all): the production's 10 stored slices (sigma/a 4; T/r_s 12 to 40; engine_fields_v14 for the dense roots) with the exterior-appended vacuum
# control on T/r_s 15, kappa 1; the sigma/a 8 member's 8 slices (engine_fields_v16); then the compose with the r_s/a 32 record (S4one/A_rs32_T18.json,
# run before this launcher with engine_fields_v12). Outputs in S4one/.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
D=H100b/states
S4=$(for t in 12 15 16 20 24 25 28 30 35 40; do printf "%s/c19_rs64_A_R2_tau8_ell4_avg_r11/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T00%s.00.npz," $D $t; done | sed 's/,$//')
S8=$(for t in 12 13 14 16 20 25 30 40; do printf "%s/c27_rs64_A_R2_tau8_ell8_prep4_avg/orbitals_c27_rs64_A_R2_tau8_ell8_prep4_avg_T00%s.00.npz," $D $t; done | sed 's/,$//')
( nice -n 10 $PY code/engine_single_object_A_v1.py --family rs64_sigma4 --run H100b/C3prod19/c19_rs64_A_R2_tau8_ell4_avg_r11 --fields-module engine_fields_v14 \
    --slices "$S4" --vacuum-control 15,1 --out-json S4one/A_rs64_sigma4.json > S4one/A_rs64_sigma4.log 2>&1; echo "sigma4 exit $?" >> S4one/status.txt ) &
( nice -n 10 $PY code/engine_single_object_A_v1.py --family rs64_sigma8 --run H100b/C3lad27/c27_rs64_A_R2_tau8_ell8_prep4_avg --fields-module engine_fields_v16 \
    --slices "$S8" --out-json S4one/A_rs64_sigma8.json > S4one/A_rs64_sigma8.log 2>&1; echo "sigma8 exit $?" >> S4one/status.txt ) &
wait
nice -n 10 $PY code/engine_single_object_A_v1.py --compose S4one/A_rs64_sigma4.json,S4one/A_rs64_sigma8.json,S4one/A_rs32_T18.json --md S4one/A_single_object_v1.md \
    --out-json S4one/A_single_object_v1_summary.json > S4one/A_compose.log 2>&1
echo "compose exit $?" >> S4one/status.txt
