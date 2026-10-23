#!/bin/bash
# Strand 3 smoke test on the box: masses, three cheap planar runs, part B, part C.
set -u
D=${QI2_ROOT}/s3_entropy
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/smoke1
mkdir -p $OUT
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
nice -n 10 $PY code/strand3_masses.py --out $OUT > $OUT/masses.log 2>&1; echo "masses exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand3_planar.py --masses $OUT/masses.json --out $OUT --workers 3 --only 16 --limit 3 > $OUT/planar.log 2>&1; echo "planar exit $?" >> $OUT/status.txt
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10 $PY code/strand3_partB.py --out $OUT > $OUT/partB.log 2>&1; echo "partB exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand3_partC.py --out $OUT --workers 16 > $OUT/partC.log 2>&1; echo "partC exit $?" >> $OUT/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
