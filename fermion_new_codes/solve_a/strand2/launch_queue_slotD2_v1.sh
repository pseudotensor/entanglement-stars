#!/bin/bash
# Slot D, second queue (builder analyst): after strand 4 Part B version 2 (slot D's first queue) writes DONE, the strand 2a red-team
# checks of SPEC_strand2a_redteam_checks.md in order A and B (one call), then C, D, E (one call each).  Two threads at most.
set -u
B=${QI2_ROOT}/solve_a
PY=${QI2_ROOT}/python
O=$B/s2a_gate/rtchecks1
mkdir -p $O
date -u '+queued (slot D2) %Y-%m-%d %H:%M:%S UTC' > $O/status.txt
until grep -q DONE $B/s4_controls/partB_run1/status_v2.txt 2>/dev/null; do sleep 30; done
cd $B/s2a_gate || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
date -u '+start AB %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
nice -n 10 $PY code/strand2a_rtchecks_AB_v1.py --out $O --gate $O/gate > $O/AB.log 2>&1
echo "AB exit $?" >> $O/status.txt
for P in C D E; do
  date -u "+start $P %Y-%m-%d %H:%M:%S UTC" >> $O/status.txt
  nice -n 10 /usr/bin/time -v $PY code/strand2a_rtchecks_CDE_v1.py --out $O --part $P > $O/$P.log 2>&1
  echo "$P exit $?" >> $O/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $O/status.txt
echo DONE >> $O/status.txt
