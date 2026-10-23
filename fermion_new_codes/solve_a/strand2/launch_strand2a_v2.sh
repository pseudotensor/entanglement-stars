#!/bin/bash
# Strand 2a gate, version 2 of the gate script (vacuum-block fix of the Fock route): pytest v2, then L2 and R4 in run2.
# C8 does not use the Fock route; its run1 output (gate script v1) is copied into run2 when it finishes.
set -u
D=${QI2_ROOT}/s2a_gate
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/run2
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
(cd code && nice -n 10 $PY -m pytest -q tests/strand2a_gate_test_v2.py > $OUT/pytest.log 2>&1); echo "pytest exit $?" >> $OUT/status.txt
for L in L2 R4; do
  nice -n 10 $PY code/strand2a_gate_v2.py --out $OUT --lattice $L > $OUT/gate_$L.log 2>&1 &
done
wait
for L in L2 R4; do echo "gate $L: $(tail -1 $OUT/gate_$L.log)" >> $OUT/status.txt; done
until grep -q "strand2a gate C8 done" $D/run1/gate_C8.log 2>/dev/null; do sleep 30; done
cp $D/run1/gate_C8.json $D/run1/gate_C8.log $OUT/ && echo "gate C8 (run1, gate v1): $(tail -1 $OUT/gate_C8.log)" >> $OUT/status.txt
nice -n 10 $PY code/strand2a_report.py --run $OUT --md $OUT/RESULT_strand2a_tables.md > $OUT/report.log 2>&1; echo "report exit $?" >> $OUT/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
