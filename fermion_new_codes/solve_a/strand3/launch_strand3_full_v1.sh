#!/bin/bash
# Strand 3 production run on the box (spec solve/specs/SPEC_strand3_entropy.md).  Threads: 13 CPU pool workers plus
# 2 single-thread GPU workers (one process per GPU) during the planar phase; 16 or fewer afterwards.
set -u
D=${QI2_ROOT}/s3_entropy
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/run1
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
nice -n 10 $PY code/strand3_masses.py --out $OUT > $OUT/masses.log 2>&1; echo "masses exit $?" >> $OUT/status.txt
# GPU 1 was occupied by another user's process at launch (nvidia-smi); GPUs 0 and 2 take the depth-64 jobs
nice -n 10 $PY code/strand3_planar_gpu.py --masses $OUT/masses.json --out $OUT --gpu 0 --slot 0 --ngpu 2 --depth 64 > $OUT/planar_gpu0.log 2>&1 &
nice -n 10 $PY code/strand3_planar_gpu.py --masses $OUT/masses.json --out $OUT --gpu 2 --slot 1 --ngpu 2 --depth 64 > $OUT/planar_gpu2.log 2>&1 &
nice -n 10 $PY code/strand3_planar.py --masses $OUT/masses.json --out $OUT --workers 14 --only 16,32 > $OUT/planar_cpu.log 2>&1; echo "planar cpu exit $?" >> $OUT/status.txt
wait; echo "planar gpu done" >> $OUT/status.txt
nice -n 10 $PY code/strand3_batches.py --masses $OUT/masses.json --out $OUT --what sphere > $OUT/sphere_batch.log 2>&1; echo "sphere exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand3_batches.py --masses $OUT/masses.json --out $OUT --what balls > $OUT/balls_batch.log 2>&1; echo "balls exit $?" >> $OUT/status.txt
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10 $PY code/strand3_partB_v2.py --out $OUT > $OUT/partB.log 2>&1; echo "partB exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand3_partC.py --out $OUT --workers 16 > $OUT/partC.log 2>&1; echo "partC exit $?" >> $OUT/status.txt
(cd code/orient && nice -n 10 $PY -c "import verify_entropy; verify_entropy.basic()" > $OUT/verify_basic.log 2>&1); echo "verify_basic exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand3_scan.py --masses $OUT/masses.json --out $OUT > $OUT/scan.log 2>&1; echo "scan exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand3_report.py --run $OUT --md $OUT/RESULT_strand3_tables.md > $OUT/report.log 2>&1; echo "report exit $?" >> $OUT/status.txt
(cd code && STRAND3_RUN=$OUT nice -n 10 $PY -m pytest -q tests/strand3_controls_test.py > $OUT/pytest.log 2>&1); echo "pytest exit $?" >> $OUT/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
