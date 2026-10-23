#!/bin/bash
# launch_replay_v9_rows_box_v1.sh (builder analyst; engine spec 37.5 (1) (b)): engine_replay_v9 --compose-rows on the box CPU (nice 10,
# 8 threads) for each run whose replay v9 handoff has been copied from pod 1 into CERT3v9/ (the pod's CERT3v9/ outputs): R_hat = R^ref +
# R^zone against engine_certificate_v3's direct residuals recomputed on the certificate's own data npz (the field modules per segment as the
# certificate ran: CERT/certificate_v3_*.json), the moments table per run (CERT3v9/rows_v9_*.md, .json). Usage: bash this <run> where <run> is
# c19 (the production: the v17 parent's handoff then the c19 child's), set17_A_tau8, set17_A_tau4, set17_A_tau16, set17_B_tau8, c27_ell4,
# c27_ell8.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
D=CERT3v9
case "$1" in
  c19) H="$D/replay9_c17_rs64_to11_handoff.npz,$D/replay9_c19_rs64_r11_handoff.npz"; J="$D/replay9_c17_rs64_to11.json,$D/replay9_c19_rs64_r11.json"
       DATA=COLL11/COLLAPSE_records_v1.1_c19_rs64_data.npz; FM="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14"; FD=engine_fields_v12 ;;
  set17_A_tau8) H="$D/replay9_set17_rs32_A_R2_tau8_ell4_avg_handoff.npz"; J="$D/replay9_set17_rs32_A_R2_tau8_ell4_avg.json"
       DATA=COLL/COLLAPSE_records_v1_set17_rs32_A_R2_tau8_ell4_avg_data.npz; FM=""; FD=engine_fields_v12 ;;
  set17_A_tau4) H="$D/replay9_set17_rs32_A_R2_tau4_ell4_avg_handoff.npz"; J="$D/replay9_set17_rs32_A_R2_tau4_ell4_avg.json"
       DATA=COLL/COLLAPSE_records_v1_set17_rs32_A_R2_tau4_ell4_avg_data.npz; FM=""; FD=engine_fields_v12 ;;
  set17_A_tau16) H="$D/replay9_set17_rs32_A_R2_tau16_ell4_avg_handoff.npz"; J="$D/replay9_set17_rs32_A_R2_tau16_ell4_avg.json"
       DATA=COLL/COLLAPSE_records_v1_set17_rs32_A_R2_tau16_ell4_avg_data.npz; FM=""; FD=engine_fields_v12 ;;
  set17_B_tau8) H="$D/replay9_set17_rs32_B_R2_tau8_ell4_R0x4_avg_handoff.npz"; J="$D/replay9_set17_rs32_B_R2_tau8_ell4_R0x4_avg.json"
       DATA=COLL/COLLAPSE_records_v1_set17_rs32_B_R2_tau8_ell4_R0x4_avg_data.npz; FM=""; FD=engine_fields_v12 ;;
  c27_ell4) H="$D/replay9_c27_rs64_ell4_prep4_handoff.npz"; J="$D/replay9_c27_rs64_ell4_prep4.json"
       DATA=COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell4_prep4_avg_data.npz; FM=""; FD=engine_fields_v16 ;;
  c27_ell8) H="$D/replay9_c27_rs64_ell8_prep4_handoff.npz"; J="$D/replay9_c27_rs64_ell8_prep4.json"
       DATA=COLL/COLLAPSE_records_v1_c27_rs64_A_R2_tau8_ell8_prep4_avg_data.npz; FM=""; FD=engine_fields_v16 ;;
  *) echo "unknown run $1"; exit 2 ;;
esac
nice -n 10 $PY code/engine_replay_v9.py --compose-rows "$H" --replay-json "$J" --data "$DATA" --fm-segments "$FM" --fm-default "$FD" \
    --md $D/rows_v9_$1.md --out-json $D/rows_v9_$1.json > $D/log_rows_v9_$1.log 2>&1
echo "rows $1 exit $? $(date -u +%H:%M:%S) UTC" >> $D/rows_status.txt
