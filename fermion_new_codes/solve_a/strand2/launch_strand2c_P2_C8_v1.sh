#!/bin/bash
# Strand 2c P2 on the cube profiles (2b cube pass 1 on library v5, c8pass1v5; blocks (1,0) and (2,0) at m 0.1, (0.2, 1), Lambda 1, k 1 and 10):
# strand2c_mismatch_v3.py --only-p2 at L = 8, 12, 16 on GPU 2, after the cube pass 1 is DONE and the 2d box work on GPU 2 (prep31_B4) is DONE
# and GPU 2 has no compute process.  One thread.
set -u
D=${QI2_ROOT}/s2c_mismatch
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/run4
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
until grep -q DONE ${QI2_ROOT}/status.txt 2>/dev/null && grep -q DONE ${QI2_ROOT}/status.txt 2>/dev/null; do sleep 60; done
U2=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
until [ -z "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$U2")" ]; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC (GPU 2 free)' >> $OUT/status.txt
nice -n 10 $PY code/strand2c_p2_profiles_v1.py --c8 ${QI2_ROOT}/c8pass1v5 --out $OUT/p2_profiles_C8.json > $OUT/profiles.log 2>&1
echo "profiles exit $?: $(tail -1 $OUT/profiles.log)" >> $OUT/status.txt
for L in 8 12 16; do
  nice -n 10 $PY code/strand2c_mismatch_v3.py --out $OUT --L $L --gpu 2 --profiles $OUT/p2_profiles_C8.json --only-p2 > $OUT/P2_L$L.log 2>&1
  echo "P2 L$L exit $?" >> $OUT/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
