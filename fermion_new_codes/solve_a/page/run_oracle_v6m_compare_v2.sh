#!/bin/bash
# run_oracle_v6m_compare_v2.sh (solve builder 7; engine spec 48.2 (3)): run_oracle_v6m_v1.sh's readings redone on its outputs (no rerun) with
# compare_oracle_v6m_v2.py (version 1 counted the input flag settings/col_diag and the follow runs' wall-clock retry counter failed_reads as
# differences); the check_coldiag_v1 readings as in v1. PASS of all writes V6M_GATE_PASS.
cd ${QI2_ROOT}/v6
PY=${QI2_ROOT}/python; O=${QI2_ROOT}/page_v6m; L=${QI2_ROOT}/page_v6l; M=$O/v6m_oracles_v2.marker
until grep -q "^end " $O/v6m_oracles.marker; do sleep 30; done
echo "start $(date -u +%H:%M:%S) UTC" > $M; ok=1
for v in off gensplit gen; do
  case $v in off) B=$O/oracle_off;; *) B=$O/oracle_${v}_diag;; esac
  $PY compare_oracle_v6m_v2.py $L/history_oracle_v6l_$v $B > $O/compare_v2_$v.txt 2>&1; tail -1 $O/compare_v2_$v.txt | grep -q PASS || ok=0
  echo "compare $v: $(head -1 $O/compare_v2_$v.txt) $(tail -1 $O/compare_v2_$v.txt)" >> $M
  if [ $v != off ]; then tail -1 $O/check_${v}.txt | grep -q PASS || ok=0; echo "check $v: $(head -1 $O/check_${v}.txt) $(tail -1 $O/check_${v}.txt)" >> $M; fi
done
echo "end $(date -u +%H:%M:%S) UTC ok $ok" >> $M
[ $ok -eq 1 ] && echo "v6m gate PASS $(date -u '+%F %T') UTC md5 $(md5sum page_history_v6m.py | cut -c1-32) (run_oracle_v6m_v1 outputs, compare_oracle_v6m_v2)" > V6M_GATE_PASS
