#!/bin/bash
# Strand 0 production run on the box (spec solve/specs/SPEC_strand0_source_allocation.md): controls tests, part A,
# the ring-L scan (T8), the cube (part B), then the report tables.  Code in ./code (data script v2).
set -u
D=${QI2_ROOT}/s0_source_allocation
cd $D || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=${QI2_ROOT}/python
OUT=$D/full_v1
mkdir -p $OUT
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
(cd code && nice -n 10 $PY -m pytest -q tests/strand0_controls_test.py > $OUT/pytest.log 2>&1); echo "pytest exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand0_partA_data_v2.py --out $OUT --stored-jobs ${QI2_ROOT}/native_jobs.tar.gz --stored-kscan ${QI2_ROOT}/native_kscan.json.gz > $OUT/partA.log 2>&1; echo "partA exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand0_ringL_onebody.py --out $OUT > $OUT/ringL.log 2>&1; echo "ringL exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand0_cube_onebody.py --out $OUT > $OUT/cube.log 2>&1; echo "cube exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand0_report.py --partA $OUT/partA.json --ringL $OUT/ringL.json --cube $OUT/cube.json --md $OUT/RESULT_strand0_tables.md > $OUT/report.log 2>&1; echo "report exit $?" >> $OUT/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
