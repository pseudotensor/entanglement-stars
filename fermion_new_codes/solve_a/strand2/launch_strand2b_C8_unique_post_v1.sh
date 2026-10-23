#!/bin/bash
# Strand 2b sec. 14 item 2, post-processing of c8unique1 (strand2b_c8_unique_post_v1.py): orbits of the noise-floor limits and the
# driver's checks per orbit; k = 1 with the reference row recomputed, then k = 1e4.  12 single-thread workers.  Output s2b_row/c8unique1post.
set -u
D=${QI2_ROOT}/s2b_row
cd $D || exit 1
mkdir -p c8unique1post
date -u '+start %Y-%m-%d %H:%M:%S UTC' > c8unique1post/status.txt
for k in 1 10000; do
  REF=""; [ "$k" = "1" ] && REF="--reference"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strand2b_c8_unique_post_v1.py --out c8unique1post --record c8pass1v5/enclosure/case_C8_b20_m0.1_gM0.2_gB1_L1_k$k.json --unique c8unique1/unique_C8_b20_m0.1_gM0.2_gB1_L1_k$k.json --workers 12 $REF > c8unique1post/post_k$k.log 2>&1
  echo "k$k exit $?" >> c8unique1post/status.txt
done
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> c8unique1post/status.txt
echo DONE >> c8unique1post/status.txt
