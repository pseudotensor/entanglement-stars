#!/bin/bash
# Strand 2b sec. 15 item 5 (strand2b_sec15_item5_v1.py): the cube's classes continued in k, and the near starts at (0.05, 0) and g = 0;
# CPU, the continuation serial (2 threads), the starts in 8 single-thread workers.  Output s2b_row/sec15item5.
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
mkdir -p sec15item5
date -u '+start %Y-%m-%d %H:%M:%S UTC' > sec15item5/status.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand2b_sec15_item5_v1.py --out sec15item5 --record c8pass1v5/enclosure/case_C8_b20_m0.1_gM0.2_gB1_L1_k1.json --post c8unique1post/post_C8_b20_m0.1_gM0.2_gB1_L1_k1.json --workers 8 > sec15item5/run.log 2>&1
echo "exit $?" >> sec15item5/status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> sec15item5/status.txt
echo DONE >> sec15item5/status.txt
