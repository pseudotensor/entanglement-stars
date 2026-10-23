#!/bin/bash
# Strand 2b sec. 14 item 2: the cube uniqueness rerun for the two enclosure cases (strand2b_c8_unique_v1.py), 8 single-thread workers,
# the k = 1 case then the k = 1e4 case.  Output s2b_row/c8unique1.
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
mkdir -p c8unique1
date -u '+start %Y-%m-%d %H:%M:%S UTC' > c8unique1/status.txt
for k in 1 10000; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand2b_c8_unique_v1.py --out c8unique1 --record c8pass1v5/enclosure/case_C8_b20_m0.1_gM0.2_gB1_L1_k$k.json --workers 8 > c8unique1/unique_k$k.log 2>&1
  echo "k$k exit $?" >> c8unique1/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> c8unique1/status.txt
echo DONE >> c8unique1/status.txt
