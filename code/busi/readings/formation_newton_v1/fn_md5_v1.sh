#!/bin/bash
# busi reading formation_newton_v1: own md5 of every file the delivery names (the final pull's 27 files, the readout files,
# the earlier mirror's run json and log, the production jsons and their manifest). Box only; nice 10; read-only on the inputs.
O=${QI2_ROOT}/formation_newton_v1; mkdir -p $O/logs; D=${QI2_ROOT}/p64_newton_h100_1
PR=${QI2_ROOT}/runs_pod7_ell4; P0=${QI2_ROOT}/v36d_from0
echo "[$(date -u +%FT%TZ)] start" > $O/logs/md5.log
{ cd $D/final && find run states logs -type f | LC_ALL=C sort | xargs nice -n 10 md5sum; } > $O/md5_final_mine.txt 2>> $O/logs/md5.log
nice -n 10 md5sum $D/readout/newton_run.json $D/readout/rc_audit_v1.json $D/readout/rc_audit_v1.md $D/readout/readout_v1.md \
  $D/run/collapse_lad_P64_s4_newton_v36nq_h100_1.json $D/logs/lad_P64_s4_newton_v36nq_h100_1.log $D/LAUNCH.txt $D/STOPPER.txt $D/args_v1.txt $D/dryparse_args.json \
  $D/ops/readout_v1.py $D/ops/MD5_code_box.txt $D/ops/MD5_code_card.txt $D/MD5_slices_verified.txt \
  $PR/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json $PR/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40_fields.npz \
  $PR/logs/c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.log \
  $P0/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0.json $P0/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_fields.npz \
  $P0/logs/c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0.log \
  ${QI2_ROOT}/MANIFEST_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0.md5 \
  ${QI2_ROOT}/rc_audit_v1.py ${QI2_ROOT}/engine_ladder_compare_v5.py \
  > $O/md5_named_mine.txt 2>> $O/logs/md5.log
diff <(LC_ALL=C sort $D/final/MD5_card.txt) <(LC_ALL=C sort $O/md5_final_mine.txt) > $O/md5_final_vs_card.diff; echo "diff rc $?" >> $O/logs/md5.log
echo "[$(date -u +%FT%TZ)] done" >> $O/logs/md5.log
