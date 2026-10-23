#!/bin/bash
# Strand 2b item 7, version 2 (strand2b_sec15_item7_v2.py: the FD Jacobian's columns over a pool of 4 single-thread workers per part): the
# cube's two classes continued in k from 5 and the uniform branch's soft mode from 2 to 10, three parts in parallel (version 1's parts were
# stopped at 14:35 UTC; its doublet part stands).  Output s2b_row/sec15item7 (item7_<part>.json, <part>_v2.log).
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
date -u '+start parts v2 %Y-%m-%d %H:%M:%S UTC' >> sec15item7/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for p in class1 class2 uniform; do
  (nice -n 10 ${QI2_ROOT}/python code/strand2b_sec15_item7_v2.py --out sec15item7 --record c8pass1v5/enclosure/case_C8_b20_m0.1_gM0.2_gB1_L1_k1.json --item5 sec15item5/item5.json --part $p --workers 4 > sec15item7/${p}_v2.log 2>&1; echo "$p v2 exit $? $(date -u +%H:%M:%S)" >> sec15item7/status.txt) &
done
wait
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> sec15item7/status.txt
echo DONE >> sec15item7/status.txt
