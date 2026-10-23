#!/bin/bash
# Strand 4 Part B.3 to B.5 version 3 (finer finite-difference steps; see strand4_partB3_v4.py's header), one thread, started beside
# the slot D2 queue in the thread freed by the cube enclosure's end (lane total 16).
set -u
D=${QI2_ROOT}/s4_controls
cd $D || exit 1
date -u '+partB3 v4 start %Y-%m-%d %H:%M:%S UTC (one thread)' >> partB_run1/status_v4.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 /usr/bin/time -v ${QI2_ROOT}/python code/strand4_partB3_v4.py --inputs $D/partB_inputs --out partB_run1 > partB_run1/partB3_v4.log 2>&1
echo "partB3 v4 exit $?" >> partB_run1/status_v4.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> partB_run1/status_v4.txt
echo DONE >> partB_run1/status_v4.txt
