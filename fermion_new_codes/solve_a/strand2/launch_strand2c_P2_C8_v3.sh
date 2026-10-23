#!/bin/bash
# Strand 2c, the cube's P2 profiles and runs (builder analyst), into run4c: slot C v5 ran this step from the wrong working directory
# (no file found; run4b/status.txt records the failed attempt, nothing computed); this launcher runs it alone, on any GPU free at
# launch, one thread on the CPU side.
set -u
D=${QI2_ROOT}/s2c_mismatch
cd $D || exit 1
PY=${QI2_ROOT}/python
OUT=$D/run4c
mkdir -p $OUT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
gpu_wait() {
  while true; do
    BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
    for i in 0 1 2; do
      Ui=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i=$i '$1==i{print $2}')
      if [ -n "$Ui" ] && ! echo "$BUSY" | grep -q "$Ui"; then GPU=$i; return 0; fi
    done
    sleep 30
  done
}
date -u '+start %Y-%m-%d %H:%M:%S UTC (profiles on the CPU)' > $OUT/status.txt
nice -n 10 $PY code/strand2c_p2_profiles_v1.py --c8 ${QI2_ROOT}/c8pass1v5 --out $OUT/p2_profiles_C8.json > $OUT/profiles.log 2>&1
echo "profiles exit $?: $(tail -1 $OUT/profiles.log)" >> $OUT/status.txt
for L in 8 12 16; do
  gpu_wait
  echo "P2 L$L on GPU $GPU $(date -u '+%H:%M:%S UTC')" >> $OUT/status.txt
  nice -n 10 $PY code/strand2c_mismatch_v3.py --out $OUT --L $L --gpu $GPU --profiles $OUT/p2_profiles_C8.json --only-p2 > $OUT/P2_L$L.log 2>&1
  echo "P2 L$L exit $?" >> $OUT/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
