#!/bin/bash
# Strand 2b sec. 15 item 3: pass 2's 45 cases rerun with driver v12 (library v7: the repaired projector denominators, the cluster gap,
# the strict line search, full start records), 12 workers (one thread each), into pass2v12/; then the cube's FD Jacobian two-step check.
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
mkdir -p pass2v12
date -u '+start %Y-%m-%d %H:%M:%S UTC (driver v12, 12 workers)' > pass2v12/status.txt
LIST=pass2v12/cases.txt
: > $LIST
for f in pass2/case_*.json; do
  b=$(basename $f .json)
  echo $b | sed -E 's/^case_R4_b([0-9])([0-9])_m([^_]+)_gM([^_]+)_gB([^_]+)_L([^_]+)_k(.+)$/\1,\2 \3 \4 \5 \6 \7/' >> $LIST
done
echo "cases: $(wc -l < $LIST)" >> pass2v12/status.txt
cat $LIST | xargs -P 12 -L 1 bash -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand2b_row_R4_v12.py --out pass2v12 --block $0 --m $1 --gM $2 --gB $3 --Lam $4 --k $5 >> pass2v12/run.log 2>&1; echo "$0 $1 $2 $3 $4 $5 exit $? $(date -u +%H:%M:%S)" >> pass2v12/status.txt'
mkdir -p c8jac2step
OMP_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand2b_c8_jac2step_v1.py --records c8pass1v5/enclosure --out c8jac2step > c8jac2step/run.log 2>&1
echo "c8jac2step exit $?" >> pass2v12/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> pass2v12/status.txt
echo DONE >> pass2v12/status.txt
