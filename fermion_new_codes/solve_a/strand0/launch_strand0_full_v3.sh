#!/bin/bash
# Strand 0 production run on the box (spec solve/specs/SPEC_strand0_source_allocation.md): controls tests, part A,
# version 3 after the reviewer B red team of strand 0: data v3, ring-L v3, cube v2, report v4 (reporting fixes; same physics objects).
set -u
D=${QI2_ROOT}/s0_source_allocation
cd $D || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
OUT=$D/full_v3
mkdir -p $OUT
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
(cd code && nice -n 10 $PY -m pytest -q tests/strand0_controls_test_v3.py > $OUT/pytest.log 2>&1); echo "pytest exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand0_partA_data_v3.py --out $OUT --stored-jobs ${QI2_ROOT}/native_jobs.tar.gz --stored-kscan ${QI2_ROOT}/native_kscan.json.gz > $OUT/partA.log 2>&1; echo "partA exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand0_ringL_onebody_v3.py --out $OUT > $OUT/ringL.log 2>&1; echo "ringL exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand0_cube_onebody_v2.py --out $OUT > $OUT/cube.log 2>&1; echo "cube exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand0_report_v4.py --partA $OUT/partA.json --ringL $OUT/ringL.json --cube $OUT/cube.json --md $OUT/RESULT_strand0_tables.md > $OUT/report.log 2>&1; echo "report exit $?" >> $OUT/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
