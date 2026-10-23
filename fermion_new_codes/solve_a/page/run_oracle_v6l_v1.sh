#!/bin/bash
# run_oracle_v6l_v1.sh (solve builder 6; engine spec 47.1 (2) (i), (v)): on the box CPU, nice 10, 8 threads, one after the other:
# (1) the gate: the oracle v6l with --extra empty, compared with the oracle v6k run 1 (compare_oracle_v6l_v1.py, every json number and npz array);
# (2) the oracle with --collector-select gen; (3) with --collector-select gen --collector-admit split. Outputs on /data (rule 9).
cd ${QI2_ROOT}/v6
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=${QI2_ROOT}/python; O=${QI2_ROOT}/page_v6l; mkdir -p $O; M=$O/v6l_oracles.marker
echo "start $(date -u +%H:%M:%S) UTC" > $M
nice -n 10 $PY page_history_oracle_v6l.py --out $O/history_oracle_v6l_off --device cpu --threads 8 > $O/history_oracle_v6l_off.log 2>&1
echo "oracle off exit $? $(date -u +%H:%M:%S) UTC" >> $M
nice -n 10 $PY compare_oracle_v6l_v1.py runs/history_oracle_v6k_run1 $O/history_oracle_v6l_off > $O/compare_oracle_v6l_v1.txt 2>&1
echo "compare exit $? $(date -u +%H:%M:%S) UTC" >> $M
nice -n 10 $PY page_history_oracle_v6l.py --out $O/history_oracle_v6l_gen --device cpu --threads 8 --extra "--collector-select gen" > $O/history_oracle_v6l_gen.log 2>&1
echo "oracle gen exit $? $(date -u +%H:%M:%S) UTC" >> $M
nice -n 10 $PY page_history_oracle_v6l.py --out $O/history_oracle_v6l_gensplit --device cpu --threads 8 --extra "--collector-select gen --collector-admit split" > $O/history_oracle_v6l_gensplit.log 2>&1
echo "oracle gensplit exit $? $(date -u +%H:%M:%S) UTC" >> $M
