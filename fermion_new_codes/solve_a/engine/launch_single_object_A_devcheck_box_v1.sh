#!/bin/bash
# launch_single_object_A_devcheck_box_v1.sh (builder analyst; the revised S4's D validation for part A, the device check): version 1 (numpy on the
# box CPU, nice 10, 8 threads) on one record of the production family that version 2 ran on pod 1 lane 2c's GPU (S4one/A_rs64_sigma4.json): the
# r_s/a 64 sigma/a 4 slice at T/r_s 15.0, kappa 1 and 2, items 1 to 5 on both roots (no vacuum control: the GPU run's control stands). The
# readouts are compared with the GPU run's by engine_single_object_A_compose_v1.py --device-check. Output S4one/A_rs64_sigma4_T15_k12_v1cpu.json.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "devcheck start $(date -u +%H:%M:%S) UTC" >> S4one/status.txt
nice -n 10 $PY code/engine_single_object_A_v1.py --family rs64_sigma4 --run H100b/C3prod19/c19_rs64_A_R2_tau8_ell4_avg_r11 --fields-module engine_fields_v14 \
    --slices H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0015.00.npz --kappas 1,2 --threads 8 \
    --out-json S4one/A_rs64_sigma4_T15_k12_v1cpu.json > S4one/A_rs64_sigma4_T15_k12_v1cpu.log 2>&1
echo "devcheck exit $? $(date -u +%H:%M:%S) UTC" >> S4one/status.txt
