#!/bin/bash
# Slot C, version 2 (builder analyst): as version 1, but the dressed vacuum runs (1), (2) start when the cube batch exits instead of also
# waiting for the 2c kernel, which only (3) needs.  Version 1's header follows.
# Slot C (builder analyst): the lane's thread cap is 16; the 2b runs hold 14 (R4 pass 2: 8; cube pass 1: 4 batch workers + 2 enclosure
# processes) and the small chain holds 2 (prep31_B4, then the 2c kernel, the strand 4 midpoint rerun, C7, the jacobian gap, strand 4
# Part B.3).  This queue takes two of the cube batch's four threads once that batch exits, and runs its jobs one at a time on GPU 2:
# (1) the 2d dressed (1,1) vacuum cost on C8 (CPU, one thread; the check against prep31dr2_C8's E11 values); (2) the same on B4 (GPU 2);
# (3) the 2c addendum parts B and C (GPU 2; replaces the waiter launch_strand2c_addBC_v1.sh, which would have started beside the small
# chain and taken the lane to 18 threads); (4) the 2c cube P2 profiles and runs (GPU 2; replaces the waiter launch_strand2c_P2_C8_v2.sh,
# which also waited for the cube enclosure processes although the profiles read only the batch's cases).  At most two threads at a time.
set -u
B=${QI2_ROOT}/solve_a
PY=${QI2_ROOT}/python
mkdir -p $B/s2d_jet/vac11 $B/s2c_mismatch/addBC1 $B/s2c_mismatch/run4b
date -u '+queued (slot C v2) %Y-%m-%d %H:%M:%S UTC' > $B/s2d_jet/vac11/status.txt
date -u '+queued (slot C v2, replaces the v1 waiter) %Y-%m-%d %H:%M:%S UTC' >> $B/s2c_mismatch/addBC1/status.txt
date -u '+queued (slot C v2, replaces the v2 waiter) %Y-%m-%d %H:%M:%S UTC' >> $B/s2c_mismatch/run4b/status.txt
until grep -q "pass1 batch exit" $B/s2b_row/c8pass1v5/status.txt 2>/dev/null; do sleep 30; done
U2=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2{print $2}')
gpu2_wait() {
  until [ -z "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep "$U2")" ]; do sleep 30; done
}
# (1) and (2)
cd $B/s2d_jet || exit 1
date -u '+start C8 %Y-%m-%d %H:%M:%S UTC' >> vac11/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/strand2d_vac11_dressed_v1.py --out vac11 --lattice C8 --gM 0.2 --gB 1 --device cpu > vac11/vac11_C8.log 2>&1
echo "C8 exit $?" >> vac11/status.txt
gpu2_wait
date -u '+start B4 %Y-%m-%d %H:%M:%S UTC (GPU 2 free)' >> vac11/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 /usr/bin/time -v $PY code/strand2d_vac11_dressed_v1.py --out vac11 --lattice B4 --gM 0.2 --gB 1 --device cuda:2 > vac11/vac11_B4.log 2>&1
echo "B4 exit $?" >> vac11/status.txt
echo DONE >> vac11/status.txt
# (3)
cd $B/s2c_mismatch || exit 1
until grep -q "kernel exit" $B/s2c_mismatch/kernel1/status.txt 2>/dev/null; do sleep 30; done
gpu2_wait
date -u '+start %Y-%m-%d %H:%M:%S UTC (GPU 2 free)' >> addBC1/status.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/strand2c_addendum_BC_v1.py --out addBC1 --profiles run3/p2_profiles_R4.json --kernel kernel1/kernel_L256.json --gpu 2 > addBC1/addBC.log 2>&1
echo "BC exit $?" >> addBC1/status.txt
echo DONE >> addBC1/status.txt
# (4)
OUT=$B/s2c_mismatch/run4b
gpu2_wait
date -u '+start %Y-%m-%d %H:%M:%S UTC (GPU 2 free)' >> $OUT/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 $PY code/strand2c_p2_profiles_v1.py --c8 $B/s2b_row/c8pass1v5 --out $OUT/p2_profiles_C8.json > $OUT/profiles.log 2>&1
echo "profiles exit $?: $(tail -1 $OUT/profiles.log)" >> $OUT/status.txt
for L in 8 12 16; do
  gpu2_wait
  nice -n 10 $PY code/strand2c_mismatch_v3.py --out $OUT --L $L --gpu 2 --profiles $OUT/p2_profiles_C8.json --only-p2 > $OUT/P2_L$L.log 2>&1
  echo "P2 L$L exit $?" >> $OUT/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> $OUT/status.txt
echo DONE >> $OUT/status.txt
