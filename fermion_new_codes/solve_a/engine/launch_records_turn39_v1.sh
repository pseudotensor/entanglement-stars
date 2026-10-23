#!/bin/bash
# The records re-headed with the lead's turn 39 labels (engine spec 31.6; FILE 3, compose_coll11_labels_v6): the production's
# COLLAPSE_records_v1.1 with label C6 (label_v6.txt; the status file unchanged, status_v5.txt) and the ell 8 record's COLLAPSE_records_v1
# with label E3 (label_c24_rs64_ell8_v3.txt), each by the composer that wrote its turn 38 / turn 37 version, the same inputs. Run on the box
# from engine/, detached. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
P=${QI2_ROOT}/python
export OMP_NUM_THREADS=4
S=COLL/status_turn39.txt
echo "records turn39 start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $P code/engine_collapse_records_v2.py --config COLL11/config_c19_rs64_v1.json --label-file COLL11/label_v6.txt --status-file COLL11/status_v5.txt --slices-dir H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11 --crosshw REPLAY2/crosshw_c17_rs64_v2.md --md COLL11/COLLAPSE_records_v1.1_c19_rs64_turn39.md --npz COLL11/COLLAPSE_records_v1.1_c19_rs64_data_turn39.npz > COLL11/log_v1.1_turn39.log 2>&1
echo "records turn39 production exit $? $(date -u +%H:%M:%S) UTC" >> $S
test ! -e COLL/COLLAPSE_records_v1_c24_rs64_A_R2_tau8_ell8_avg_turn39.md && nice -n 10 $P code/engine_collapse_records_v1.py --run H100b/C3prod24/c24_rs64_A_R2_tau8_ell8_avg --handoff H100b/C3prod24/collapse_c24_rs64_A_R2_tau8_ell8_avg_handoff.npz --spectra H100b/C3prod24/collapse_c24_rs64_A_R2_tau8_ell8_avg_spectra.npz --rows ROWS/rows_c24_rs64_ell8_v4.json --replay-json REPLAY6/replay7_c24_rs64_ell8.json --label "$(cat COLL/label_c24_rs64_ell8_v3.txt)" --md COLL/COLLAPSE_records_v1_c24_rs64_A_R2_tau8_ell8_avg_turn39.md --npz COLL/COLLAPSE_records_v1_c24_rs64_A_R2_tau8_ell8_avg_data_turn39.npz > COLL/log_records_c24_ell8_v3.log 2>&1
echo "records turn39 ell8 exit $? $(date -u +%H:%M:%S) UTC" >> $S
md5sum COLL11/COLLAPSE_records_v1.1_c19_rs64_data_turn38.npz COLL11/COLLAPSE_records_v1.1_c19_rs64_data_turn39.npz COLL/COLLAPSE_records_v1_c24_rs64_A_R2_tau8_ell8_avg_data_turn37.npz COLL/COLLAPSE_records_v1_c24_rs64_A_R2_tau8_ell8_avg_data_turn39.npz >> $S 2>&1
echo "records turn39 DONE $(date -u +%H:%M:%S) UTC" >> $S
