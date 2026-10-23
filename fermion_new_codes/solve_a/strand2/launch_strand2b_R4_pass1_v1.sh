#!/bin/bash
# Strand 2b R4 production, pass 1: every law case (6 blocks x 3 m x 4 coupling rows x 5 k = 360) solved by continuation with the
# lock pieces, without the enclosure and the uniqueness scan (pass 2 adds them on the lead's choice of scope); then the native
# control (sec. 11.3) and the ruler variances.  12 single-thread workers (2a's C8 run and 2c's L = 16 run hold the rest).
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/pass1
mkdir -p $OUT/native
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+start %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
nice -n 10 $PY code/strand2b_batch_R4_v8.py --out $OUT --workers 12 --no-enclosure --no-unique > $OUT/batch.log 2>&1; echo "law batch exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand2b_native_R4_v1.py --out $OUT/native --variance ${QI2_ROOT}/mats > $OUT/native_variance.log 2>&1; echo "variance exit $?" >> $OUT/status.txt
for q in 0 1 2; do for m in 0.1 0.2 0.4; do for U in 0 0.1 0.4; do for k in 1 10 100 1000 10000; do
  echo "$q $m $U $k"
done; done; done; done > $OUT/native_cases.txt
cat $OUT/native_cases.txt | xargs -P 12 -L 1 bash -c 'nice -n 10 '"$PY"' code/strand2b_native_R4_v1.py --out '"$OUT"'/native --q $0 --m $1 --U $2 --k $3 --no-enclosure --no-unique >> '"$OUT"'/native/native.log 2>&1'
echo "native batch done" >> $OUT/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
