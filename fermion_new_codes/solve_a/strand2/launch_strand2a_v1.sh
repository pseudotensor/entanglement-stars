#!/bin/bash
# Strand 2a gate on the box (spec solve/specs/SPEC_strand2a_gate.md): pytest of G1-G6 on small cases, then the three
# lattices in parallel (3 single-thread processes), then the report.
set -u
D=${QI2_ROOT}/s2a_gate
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/run1
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
(cd code && nice -n 10 $PY -m pytest -q tests/strand2a_gate_test.py > $OUT/pytest.log 2>&1); echo "pytest exit $?" >> $OUT/status.txt
for L in L2 R4 C8; do
  nice -n 10 $PY code/strand2a_gate.py --out $OUT --lattice $L > $OUT/gate_$L.log 2>&1 &
done
wait
for L in L2 R4 C8; do echo "gate $L: $(tail -1 $OUT/gate_$L.log)" >> $OUT/status.txt; done
nice -n 10 $PY code/strand2a_report.py --run $OUT --md $OUT/RESULT_strand2a_tables.md > $OUT/report.log 2>&1; echo "report exit $?" >> $OUT/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
