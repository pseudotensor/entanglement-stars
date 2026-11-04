#!/bin/bash
# run_page_v4_box_v1b.sh (the solve lane; PAGE_solved_history_v4.1): the box run lines. Everything runs on compute-host under nice 10 with at most 16 threads;
# the scripts read stored files only (no propagation, no GPU). The s8 K = 18 and the from-14 post-pass runs (the busi lane's w1_postpass_v1f.py) were
# launched the same way before this script (their logs are in $B/logs); they are listed here for the record.
set -e
B=${QI2_ROOT}/page_v4; C=$B/code; O=$B/out_v41; mkdir -p $O $B/logs
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 W1_CODE_D13=${QI2_ROOT}/code_d13 W1_CODE_V6=${QI2_ROOT}/code_v6 MPLBACKEND=Agg
# (for the record) the post-pass reruns:
#   PP=${QI2_ROOT}/w1_postpass_v1f.py; ST1=${QI2_ROOT}/w1_selftest_v1.json; ST2=${QI2_ROOT}/w1_selftest_v1.json
#   nice -n 10 $PY $PP --prod $B/s8_prod_K18 v6o_s8 --out $B/postpass/W1_s8_v1f_K18_from40 --selftest $ST1 --selftest $ST2 --tol 0.01 --from-rs 40 --kappa-max 18
#   nice -n 10 $PY $PP --prod $B/s8_prod_K18 v6o_s8 --out $B/postpass/W1_s8_v1f_K18_from14 --selftest $ST1 --selftest $ST2 --tol 0.01 --from-rs 14 --kappa-max 18
#   nice -n 10 $PY $PP --prod ${QI2_ROOT}/prod v6o_B2 --ctrl ${QI2_ROOT}/v6n_B2_pod7_k1to8 v6n_B2 --out $B/postpass/W1_B2_v1f_K18_from14 --selftest $ST1 --selftest $ST2 --tol 0.01 --from-rs 14 --kappa-max 18 --sentences-b2
nice -n 10 $PY $C/page_v4_data_v1b.py $O > $B/logs/page_v4_data_v1b.log 2>&1
nice -n 10 $PY $C/page_v4_tables_v1b.py $O/page_v4_data_v1.json $O/page_v4_tables_v1b.md $O/page_v4_readout_export_v1b.tsv > $B/logs/page_v4_tables_v1b.log 2>&1
nice -n 10 $PY $C/page_v4_fig_v1b.py $O/page_v4_data_v1.json $O/page_v4_data_v1.npz $O/PAGE_fig_v4.1.png $O/PAGE_fig_v4.1_data.npz > $B/logs/page_v4_fig_v1b.log 2>&1
cd $O && md5sum page_v4_data_v1.json page_v4_data_v1.npz page_v4_tables_v1b.md page_v4_readout_export_v1b.tsv PAGE_fig_v4.1.png PAGE_fig_v4.1_data.npz > OUTPUTS.md5
tail -n 3 $B/logs/page_v4_data_v1b.log $B/logs/page_v4_tables_v1b.log $B/logs/page_v4_fig_v1b.log; cat $O/OUTPUTS.md5
