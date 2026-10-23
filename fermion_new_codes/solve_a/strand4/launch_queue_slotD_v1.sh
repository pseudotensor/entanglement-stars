#!/bin/bash
# Slot D (builder analyst): the second pair of the cube pass-1 batch's four threads, free once that batch exits (slot C holds the other
# pair).  Runs strand 4 Part B.3 to B.5 version 2 (the lead's ruling of 07:13 UTC; replaces the version-1 waiter launch_strand4_partB3_v1.sh,
# which would have run in the small chain after the jacobian-gap check).  Two threads; writes DONE to partB_run1/status_v2.txt.
set -u
B=${QI2_ROOT}/solve_a
PY=${QI2_ROOT}/python
D=$B/s4_controls
mkdir -p $D/partB_run1
date -u '+queued (slot D) %Y-%m-%d %H:%M:%S UTC' > $D/partB_run1/status_v2.txt
until grep -q "pass1 batch exit" $B/s2b_row/c8pass1v5/status.txt 2>/dev/null; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $D/partB_run1/status_v2.txt
cd $D || exit 1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 /usr/bin/time -v $PY code/strand4_partB3_v2.py --inputs $D/partB_inputs --out partB_run1 > partB_run1/partB3_v2.log 2>&1
echo "partB3 v2 exit $?" >> $D/partB_run1/status_v2.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $D/partB_run1/status_v2.txt
echo DONE >> $D/partB_run1/status_v2.txt
