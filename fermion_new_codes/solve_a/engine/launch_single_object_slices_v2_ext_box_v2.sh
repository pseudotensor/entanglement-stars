#!/bin/bash
# launch_single_object_slices_v2_ext_box_v2.sh (builder analyst, the HORIZON version 2 builder; engine spec 39.5 (1) (vi), (vii), (ix)):
# version 1 of this launcher (launch_single_object_slices_v2_ext_box_v1.sh, waiting on the S11 gates' G1) was stopped by its exact PID at
# 02:11 UTC before it ran anything, so that the builder's box jobs stay at 8 CPU threads; this version waits until the B2 retry chain has ended
# (S4one/v2/status.txt "B2 retry all done"), then runs the same call: engine_single_object_slices_v2.py on #36 B's continuation's stored slices
# T/r_s 40, 42, 44 and 46 against the snapshot S4one/v2/inputs/ext_snap1/ (md5-checked at 01:58:20 UTC), box GPU 0, nice 10, 2 CPU threads.
# Output S4one/v2/slices_v2_ext_b.json. Usage (from ${QI2_ROOT}/engine): nohup bash code/launch_single_object_slices_v2_ext_box_v2.sh &
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PAGE_V6_PATH=${QI2_ROOT}/v6
PY=${QI2_ROOT}/python
S=S4one/v2/status.txt
D=S4one/v2/inputs/ext_snap1
DE=states/c19_rs64_A_R2_tau8_ell4_avg_ext100
until grep -q "B2 retry all done" $S 2>/dev/null; do sleep 30; done
L=$DE/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0040.00.npz,$DE/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0042.00.npz,$DE/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0044.00.npz,$DE/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0046.00.npz
echo "slices_v2_ext_b start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_single_object_slices_v2.py --name continuation --record $D/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json \
    --fields $D/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz --spectra $D/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_spectra.npz \
    --fields-module engine_fields_v14 --slices $L --kappas 1,2 --gpu 0 --out-json S4one/v2/slices_v2_ext_b.json > S4one/v2/slices_v2_ext_b.log 2>&1
echo "slices_v2_ext_b exit $? $(date -u +%H:%M:%S) UTC" >> $S
