#!/bin/bash
# run_oracle_v6m_v1.sh (solve builder 7; engine spec 48.2 (3)): check (viii)'s code gated as v6l's was, on the box CPU, nice 10, 8 threads:
# (1) the oracle v6m with --extra empty against the oracle v6l off (compare_oracle_v6l_v1.py: every json number and npz array);
# (2) with '--collector-select gen --collector-admit split --col-diag' against the oracle v6l gensplit (compare_oracle_v6m_v1.py: v6l's numbers
#     unchanged, col_diag excluded) and its col_diag entries read by check_coldiag_v1.py (bounds);
# (3) with '--collector-select gen --col-diag' against the oracle v6l gen, the same two readings.
# PASS of all writes ${QI2_ROOT}/V6M_GATE_PASS (the queues' start marker). Outputs on /data (rule 9).
cd ${QI2_ROOT}/v6
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=${QI2_ROOT}/python; O=${QI2_ROOT}/page_v6m; L=${QI2_ROOT}/page_v6l; mkdir -p $O; M=$O/v6m_oracles.marker
echo "start $(date -u +%H:%M:%S) UTC" > $M; ok=1
nice -n 10 $PY page_history_oracle_v6m.py --out $O/oracle_off --device cpu --threads 8 > $O/oracle_off.log 2>&1; echo "oracle off exit $?" >> $M
nice -n 10 $PY compare_oracle_v6l_v1.py $L/history_oracle_v6l_off $O/oracle_off > $O/compare_off.txt 2>&1; tail -1 $O/compare_off.txt | grep -q PASS || ok=0; echo "compare off: $(head -1 $O/compare_off.txt) $(tail -1 $O/compare_off.txt)" >> $M
for v in gensplit gen; do
  case $v in gensplit) X="--collector-select gen --collector-admit split --col-diag";; gen) X="--collector-select gen --col-diag";; esac
  nice -n 10 $PY page_history_oracle_v6m.py --out $O/oracle_${v}_diag --device cpu --threads 8 --extra "$X" > $O/oracle_${v}_diag.log 2>&1; echo "oracle $v diag exit $?" >> $M
  nice -n 10 $PY compare_oracle_v6m_v1.py $L/history_oracle_v6l_$v $O/oracle_${v}_diag > $O/compare_${v}.txt 2>&1; tail -1 $O/compare_${v}.txt | grep -q PASS || ok=0
  echo "compare $v (col_diag excluded): $(head -1 $O/compare_${v}.txt) $(tail -1 $O/compare_${v}.txt)" >> $M
  nice -n 10 $PY check_coldiag_v1.py $O/oracle_${v}_diag --expect-entries > $O/check_${v}.txt 2>&1; tail -1 $O/check_${v}.txt | grep -q PASS || ok=0
  echo "check $v: $(head -1 $O/check_${v}.txt) $(tail -1 $O/check_${v}.txt)" >> $M
done
echo "end $(date -u +%H:%M:%S) UTC ok $ok" >> $M
[ $ok -eq 1 ] && { echo "v6m gate PASS $(date -u '+%F %T') UTC md5 $(md5sum page_history_v6m.py | cut -c1-32)" > V6M_GATE_PASS; }
