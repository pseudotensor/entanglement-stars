#!/bin/bash
# The lead's small CPU jobs, run in the two-thread slot that the 2d box jet and its stage-1 run hold (the 2b runs hold the other 14):
# waits for prep31_B4 DONE, then sequentially: 2c addendum part A (strand2c_kernel_v1.py: the dense control at L 8, 12 and the kernel
# on L = 256); strand 4 Part A with the midpoint hopping weight (launch_strand4_partA_mid_v1.sh); 2d control C7 on C8 and B4
# (strand2d_c7_v1.py).  Two threads.
set -u
B=${QI2_ROOT}/solve_a
PY=${QI2_ROOT}/python
mkdir -p $B/s2c_mismatch/kernel1 $B/s2d_jet/c7
echo "$(date -u '+queued %Y-%m-%d %H:%M:%S UTC')" > $B/s2c_mismatch/kernel1/status.txt
until grep -q DONE $B/s2d_jet/prep31_B4/status.txt 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "$(date -u '+start %Y-%m-%d %H:%M:%S UTC')" >> $B/s2c_mismatch/kernel1/status.txt
cd $B/s2c_mismatch && nice -n 10 $PY code/strand2c_kernel_v1.py --out kernel1 --control --L 256 > kernel1/kernel.log 2>&1; echo "kernel exit $?" >> kernel1/status.txt
cd $B/s4_controls && bash launch_strand4_partA_mid_v1.sh
cd $B/s2d_jet && nice -n 10 $PY code/strand2d_c7_v1.py --out c7 --lattice C8 --jet C8run2/jet_C8_gM0_gB0.json --prep31 prep31_C8/prep31_C8_gM0_gB0.json > c7/c7_C8.log 2>&1; echo "c7 C8 exit $?" >> c7/status.txt
nice -n 10 $PY code/strand2d_c7_v1.py --out c7 --lattice B4 --jet B4run1/jet_B4_gM0_gB0.json --prep31 prep31_B4/prep31_B4_gM0_gB0.json > c7/c7_B4.log 2>&1; echo "c7 B4 exit $?" >> c7/status.txt
echo DONE >> c7/status.txt
echo "$(date -u '+end %Y-%m-%d %H:%M:%S UTC')" >> $B/s2c_mismatch/kernel1/status.txt
