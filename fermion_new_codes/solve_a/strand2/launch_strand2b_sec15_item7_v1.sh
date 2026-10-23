#!/bin/bash
# Strand 2b item 7 (strand2b_sec15_item7_v1.py): the cube's two classes continued in k from 5 and the uniform branch's soft mode from 2 to
# 10, three single-thread processes, after strand 2b item 6's R4 workers are done (thread budget); the doublet part ran first (14:00 UTC).
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
until grep -q "R4 cases done" pass2v13/status.txt 2>/dev/null; do sleep 30; done
date -u '+start parts %Y-%m-%d %H:%M:%S UTC' >> sec15item7/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for p in class1 class2 uniform; do
  (nice -n 10 ${QI2_ROOT}/python code/strand2b_sec15_item7_v1.py --out sec15item7 --record c8pass1v5/enclosure/case_C8_b20_m0.1_gM0.2_gB1_L1_k1.json --item5 sec15item5/item5.json --part $p > sec15item7/$p.log 2>&1; echo "$p exit $? $(date -u +%H:%M:%S)" >> sec15item7/status.txt) &
done
wait
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> sec15item7/status.txt
echo DONE >> sec15item7/status.txt
