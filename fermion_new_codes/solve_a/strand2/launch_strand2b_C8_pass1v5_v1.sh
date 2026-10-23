#!/bin/bash
# Strand 2b C8 pass 1 (SPEC sec. 12(b)) on library v5 (batch v4 -> driver v5): blocks (1,0) and (2,0), m 0.1 and 0.2, the coupling rows
# (0, 0, 1), (0.05, 0, 1), (0.2, 1, 1), k 1, 10, 100, 1e4 (48 cases), no enclosure or uniqueness, 4 single-thread workers; then the
# enclosure and uniqueness of sec. 12(b) (driver v6: radii 1e-3, 1e-2, 1e-1; 16 axes + 24 random; 8 starts) on (2,0), m 0.1, (0.2, 1, 1),
# k 1 and 1e4, two single-thread processes.  Starts when strand 3 addendum part A has finished (16-thread budget).
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/c8pass1v5
mkdir -p $OUT/enclosure
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
until grep -q "A exit" ${QI2_ROOT}/status.txt 2>/dev/null; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
for k in 1 10000; do
  (nice -n 10 $PY code/strand2b_row_C8_v6.py --out $OUT/enclosure --block 2,0 --m 0.1 --gM 0.2 --gB 1 --Lam 1 --k $k --enclosure --unique > $OUT/enclosure/case_k$k.log 2>&1; echo "enclosure k$k exit $?" >> $OUT/status.txt) &
done
nice -n 10 $PY code/strand2b_batch_C8_v4.py --out $OUT --workers 4 --blocks "1,0;2,0" --ms 0.1,0.2 --couplings 0,1,2 --ks 1,10,100,10000 --no-enclosure --no-unique > $OUT/batch.log 2>&1; echo "pass1 batch exit $?" >> $OUT/status.txt
wait
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
