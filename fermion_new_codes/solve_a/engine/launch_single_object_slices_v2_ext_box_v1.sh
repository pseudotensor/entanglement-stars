#!/bin/bash
# launch_single_object_slices_v2_ext_box_v1.sh (builder analyst, the HORIZON version 2 builder; engine spec 39.5 (1) (vi), (vii), (ix)):
# engine_single_object_slices_v2.py on #36 B's continuation's stored slices T/r_s 40, 42, 44 and 46, against a snapshot of the continuation's
# record json, fields npz and spectra npz (S4one/v2/inputs/ext_snap1/, copied md5-checked at 01:58:20 UTC: the live run rewrites its files
# every 5 records, which broke the first call's lazy reads at T/r_s 42, slices_v2_ext_a). Box GPU 0, nice 10, 2 CPU threads; it starts when
# the S11 gates' G1 has been compared (S11cert/status.txt), so that the builder's box jobs stay at 8 threads. Output S4one/v2/slices_v2_ext_b.json.
# Usage (from ${QI2_ROOT}/engine): nohup bash code/launch_single_object_slices_v2_ext_box_v1.sh &
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PAGE_V6_PATH=${QI2_ROOT}/v6
PY=${QI2_ROOT}/python
S=S4one/v2/status.txt
D=S4one/v2/inputs/ext_snap1
DE=states/c19_rs64_A_R2_tau8_ell4_avg_ext100
until grep -q "G1 compared" S11cert/status.txt 2>/dev/null; do sleep 30; done
L=$DE/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0040.00.npz,$DE/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0042.00.npz,$DE/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0044.00.npz,$DE/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0046.00.npz
echo "slices_v2_ext_b start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_single_object_slices_v2.py --name continuation --record $D/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json \
    --fields $D/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz --spectra $D/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_spectra.npz \
    --fields-module engine_fields_v14 --slices $L --kappas 1,2 --gpu 0 --out-json S4one/v2/slices_v2_ext_b.json > S4one/v2/slices_v2_ext_b.log 2>&1
echo "slices_v2_ext_b exit $? $(date -u +%H:%M:%S) UTC" >> $S
