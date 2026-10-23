#!/bin/bash
# The production's records re-headed with the lead's turn 40 labels (engine spec 32.6; FILE 4, compose_coll11_labels_v7: C7 appended to
# label C, F1 opening label F): COLLAPSE_records_v1.1 by engine_collapse_records_v2 with label_v7.txt (the status file unchanged,
# status_v5.txt), the same inputs as turn 39's (launch_records_turn39_v1.sh), the ell 8 record not re-headed this turn. Run on the box from
# engine/, detached. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
P=${QI2_ROOT}/python
export OMP_NUM_THREADS=4
S=COLL/status_turn40.txt
echo "records turn40 start $(date -u +%H:%M:%S) UTC" >> $S
nice -n 10 $P code/engine_collapse_records_v2.py --config COLL11/config_c19_rs64_v1.json --label-file COLL11/label_v7.txt --status-file COLL11/status_v5.txt --slices-dir H100b/states/c19_rs64_A_R2_tau8_ell4_avg_r11 --crosshw REPLAY2/crosshw_c17_rs64_v2.md --md COLL11/COLLAPSE_records_v1.1_c19_rs64_turn40.md --npz COLL11/COLLAPSE_records_v1.1_c19_rs64_data_turn40.npz > COLL11/log_v1.1_turn40.log 2>&1
echo "records turn40 production exit $? $(date -u +%H:%M:%S) UTC" >> $S
md5sum COLL11/COLLAPSE_records_v1.1_c19_rs64_data_turn39.npz COLL11/COLLAPSE_records_v1.1_c19_rs64_data_turn40.npz >> $S 2>&1
echo "records turn40 DONE $(date -u +%H:%M:%S) UTC" >> $S

