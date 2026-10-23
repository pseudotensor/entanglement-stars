#!/bin/bash
# Strand 2c P2 on the ring4 profiles (2b R4 pass 1 solutions; blocks (1,0), (2,1), (1,1) at m 0.1, (0.2, 1), Lambda 1, k 1 and 10):
# strand2c_mismatch_v3.py --only-p2 at L = 8, 12, 16 on GPU 0 once 2c run2 (L = 16 on GPU 0) is DONE and GPU 0 has no compute
# process (GPU 1 is another user's; GPU 2 held another lane's job at this launcher's writing).  One thread (eigh on the GPU).
# The C8 profiles follow in a separate run when 2b's C8 cases exist.  (Version 1 of this launcher, for GPU 2, never ran.)
set -u
D=${QI2_ROOT}/s2c_mismatch
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/run3
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
date -u '+queued %Y-%m-%d %H:%M:%S UTC' > $OUT/status.txt
until grep -q DONE $D/run2/status.txt 2>/dev/null; do sleep 30; done
U0=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==0{print $2}')
until [ -z "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$U0")" ]; do sleep 30; done
date -u '+start %Y-%m-%d %H:%M:%S UTC (GPU 0 free)' >> $OUT/status.txt
nice -n 10 $PY code/strand2c_p2_profiles_v1.py --r4 ${QI2_ROOT}/pass1 --out $OUT/p2_profiles_R4.json > $OUT/profiles.log 2>&1
echo "profiles exit $?: $(tail -1 $OUT/profiles.log)" >> $OUT/status.txt
for L in 8 12 16; do
  nice -n 10 $PY code/strand2c_mismatch_v3.py --out $OUT --L $L --gpu 0 --profiles $OUT/p2_profiles_R4.json --only-p2 > $OUT/P2_L$L.log 2>&1
  echo "P2 L$L exit $?" >> $OUT/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
