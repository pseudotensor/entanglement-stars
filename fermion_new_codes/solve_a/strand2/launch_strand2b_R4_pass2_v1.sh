#!/bin/bash
# Strand 2b R4 pass 2 (SPEC_strand2b_row sec. 12(a)): enclosure and 20-start uniqueness on 45 law cases (blocks (1,1), (2,1) at
# (0.05, 0, 1) and (0.2, 1, 1), and (1,0) at g = 0; m 0.1, 0.2, 0.4; k 1, 10, 1e4) with driver v9 (batch v9), 8 single-thread
# workers; then the native control's uniqueness scan at q = 1, 2, U = 0.4, m = 0.1, k = 1, 1e4 (4 cases).  Starts when strand 3
# addendum part A (12 workers) has finished, to keep this lane at 16 threads.
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/pass2
mkdir -p $OUT/native
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
until grep -q "A exit" ${QI2_ROOT}/status.txt 2>/dev/null; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
nice -n 10 $PY code/strand2b_batch_R4_v9.py --out $OUT --workers 8 --blocks "1,1;2,1" --couplings "1,2" --ms 0.1,0.2,0.4 --ks 1,10,10000 > $OUT/batch_dressed.log 2>&1; echo "dressed batch exit $?" >> $OUT/status.txt
nice -n 10 $PY code/strand2b_batch_R4_v9.py --out $OUT --workers 8 --blocks "1,0" --couplings "0" --ms 0.1,0.2,0.4 --ks 1,10,10000 > $OUT/batch_ref.log 2>&1; echo "reference batch exit $?" >> $OUT/status.txt
for q in 1 2; do for k in 1 10000; do echo "$q $k"; done; done | xargs -P 4 -L 1 bash -c 'nice -n 10 '"$PY"' code/strand2b_native_R4_v1.py --out '"$OUT"'/native --q $0 --m 0.1 --U 0.4 --k $1 --no-enclosure >> '"$OUT"'/native/native.log 2>&1'
echo "native done" >> $OUT/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
