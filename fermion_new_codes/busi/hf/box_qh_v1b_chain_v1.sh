#!/bin/bash
# box_qh_v1b_chain_v1.sh (busi lane, WEB40G busi 6): RUN_ORDER_QH_v1b.md steps 1 to 5 in order on the box CPU, each step started only while
# the 1-minute load is below 90 (the load written at each start), nice 10, 16 threads. Box paths throughout (the worktree is not on the box;
# star_var_v3_chan was copied to ${QI2_ROOT}/star_var_v3_chan). Every stage is resumable per member (the script skips
# a member whose json exists; the channel loop restarts from its npz checkpoint), so a rerun of this chain continues where it stopped.
cd ${QI2_ROOT}/hf || exit 1
PY=${QI2_ROOT}/python; OUT=${QI2_ROOT}/qh_v1b
STAR=${QI2_ROOT}/star_m0.1_g0.25_ell8.json; ENG=${QI2_ROOT}/code; S10CODE=${QI2_ROOT}/hf
S10R="${QI2_ROOT}/star_var_v2_e_pod ${QI2_ROOT}/star_var_v3_cert ${QI2_ROOT}/star_var_v3_chan"
ST=$OUT/chain_status.txt; mkdir -p $OUT
gate() { while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -ge 90 ]; do sleep 120; done; echo "$(date -u +%FT%TZ) start $1 load $(cut -d' ' -f1 /proc/loadavg)" >> $ST; }
done_() { echo "$(date -u +%FT%TZ) end $1 rc $2" >> $ST; }
gate step1; nice -n 10 $PY qh_branch_witness_v1b.py --star $STAR --members 7,1 --stage witness --engine-dir $ENG --s10-dir $S10CODE --s10-results $S10R --out $OUT --exact-doubles-rank 16 --threads 16 >> $OUT/qh_v1b_witness_m7_m1.log 2>&1; done_ step1 $?
for m in 7 1; do gate step2_m$m; nice -n 10 $PY qh_branch_witness_v1b.py --star $STAR --members $m --stage channels --engine-dir $ENG --s10-dir $S10CODE --out $OUT --threads 16 >> $OUT/qh_v1b_channels_m$m.log 2>&1; done_ step2_m$m $?; done
gate step3; nice -n 10 $PY qh_branch_witness_v1b.py --star $STAR --members closed --stage witness --engine-dir $ENG --s10-dir $S10CODE --s10-results $S10R --out $OUT --threads 16 >> $OUT/qh_v1b_witness_closed.log 2>&1; done_ step3 $?
for m in 3 5 17 15 13 11 9; do gate step4_m$m; nice -n 10 $PY qh_branch_witness_v1b.py --star $STAR --members $m --stage channels --engine-dir $ENG --s10-dir $S10CODE --out $OUT --threads 16 >> $OUT/qh_v1b_channels_m$m.log 2>&1; done_ step4_m$m $?; done
gate step5; nice -n 10 $PY qh_branch_witness_v1b.py --stage table --out $OUT >> $OUT/qh_v1b_table.log 2>&1; done_ step5 $?
