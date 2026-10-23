#!/bin/bash
# launch_single_object_slices_v2_box_v1.sh (builder analyst, the HORIZON version 2 builder; engine spec 39.5 (1) (vi), (vii), (ix)):
# engine_single_object_slices_v2.py on box GPU 0 (nice 10, 4 CPU threads beside the B v2 job's 4), kappa 1 and 2, in turn:
# (1) the production's ten stored slices (T/r_s 12, 15, 16, 20, 24, 25, 28, 30, 35, 40; C3prod19's record, fields and spectra);
# (2) #36 B's continuation's slices present at the start (C3ext19; T/r_s 40, 42, 44, ... every 2 r_s as they land; later ones by a new call);
# (3) the sigma/a 8 history's eight slices (T/r_s 12, 13, 14, 16, 20, 25, 30, 40; its spectra copied from pod 1 md5-checked to S4one/v2/inputs/).
# Outputs in S4one/v2/; status lines in S4one/v2/status.txt. Usage (from ${QI2_ROOT}/engine): bash code/launch_single_object_slices_v2_box_v1.sh
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PAGE_V6_PATH=${QI2_ROOT}/v6
PY=${QI2_ROOT}/python
S=S4one/v2/status.txt
DP=H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11
DE=states/c19_rs64_A_R2_tau8_ell4_avg_ext100
D8=H100b/states/c27_rs64_A_R2_tau8_ell8_prep4_avg
lst() { find $1 -maxdepth 1 -name 'orbitals_*_T*.npz' -mmin +3 | sort | tr '\n' ',' | sed 's/,$//'; }   # a slice written in the last 3 minutes may be incomplete: left for a later call
echo "slices_v2_prod start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_single_object_slices_v2.py --name production --record H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json \
    --fields H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz --spectra H100b/C3prod19/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_spectra.npz \
    --fields-module engine_fields_v14 --slices $(lst $DP) --kappas 1,2 --gpu 0 --out-json S4one/v2/slices_v2_prod.json > S4one/v2/slices_v2_prod.log 2>&1
echo "slices_v2_prod exit $? $(date -u +%H:%M:%S) UTC" >> $S
echo "slices_v2_ext start $(date -u +%H:%M:%S) UTC ($(lst $DE))" >> $S
nice -n 10 $PY code/engine_single_object_slices_v2.py --name continuation --record C3ext19/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100.json \
    --fields C3ext19/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_fields.npz --spectra C3ext19/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext100_spectra.npz \
    --fields-module engine_fields_v14 --slices $(lst $DE) --kappas 1,2 --gpu 0 --out-json S4one/v2/slices_v2_ext_a.json > S4one/v2/slices_v2_ext_a.log 2>&1
echo "slices_v2_ext exit $? $(date -u +%H:%M:%S) UTC" >> $S
echo "slices_v2_s8 start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $PY code/engine_single_object_slices_v2.py --name sigma8 --record H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json \
    --fields H100b/C3lad27/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz --spectra S4one/v2/inputs/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_spectra.npz \
    --fields-module engine_fields_v16 --slices $(lst $D8) --kappas 1,2 --gpu 0 --out-json S4one/v2/slices_v2_s8.json > S4one/v2/slices_v2_s8.log 2>&1
echo "slices_v2_s8 exit $? $(date -u +%H:%M:%S) UTC" >> $S
