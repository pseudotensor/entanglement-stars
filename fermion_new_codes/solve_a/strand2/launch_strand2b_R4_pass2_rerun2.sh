#!/bin/bash
# Strand 2b R4 pass 2, rerun 2 (builder analyst): the ten further pass 2 records that ended with an exception under driver v9
# (LinAlgError "Eigenvalues did not converge", IndexError, ValueError "A^dag A not positive definite" at uniqueness starts or enclosure
# points), rerun with driver v11 (any exception at such a point recorded, not fatal).  The v9 error records move to pass2/errors/.
# Six workers, one thread each.
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
mkdir -p pass2/errors
date -u '+rerun2 start %Y-%m-%d %H:%M:%S UTC (driver v11, 6 workers)' >> pass2/rerun2_status.txt
LIST=pass2/rerun2_cases.txt
: > $LIST
for f in pass2/case_*.json; do
  if grep -q '"error"' $f; then
    b=$(basename $f .json)
    mv $f pass2/errors/${b}_v9_error.json
    # case_R4_b<blk>_m<m>_gM<gM>_gB<gB>_L<L>_k<k>
    echo $b | sed -E 's/^case_R4_b([0-9])([0-9])_m([^_]+)_gM([^_]+)_gB([^_]+)_L([^_]+)_k(.+)$/\1,\2 \3 \4 \5 \6 \7/' >> $LIST
  fi
done
echo "cases: $(wc -l < $LIST)" >> pass2/rerun2_status.txt
cat $LIST | xargs -P 6 -L 1 bash -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand2b_row_R4_v11.py --out pass2 --block $0 --m $1 --gM $2 --gB $3 --Lam $4 --k $5 >> pass2/rerun2.log 2>&1; echo "$0 $1 $2 $3 $4 $5 exit $?" >> pass2/rerun2_status.txt'
date -u '+rerun2 end %Y-%m-%d %H:%M:%S UTC' >> pass2/rerun2_status.txt
echo DONE >> pass2/rerun2_status.txt
