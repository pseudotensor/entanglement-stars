#!/bin/bash
# run_n12_v1b.sh: the v1b rerun (conjugation fix in the stress-coefficient comparison, spin vertices added; N2 with ell = 2, 3, 4, 6).
set -e
PY=${QI2_ROOT}/python
OUT=${QI2_ROOT}/n12
cd "$(dirname "$0")"
while [ ! -f $OUT/DONE_n12_v1 ]; do sleep 20; done
nice -n 10 $PY n2_retained_gram_v1b.py --out $OUT > $OUT/n2_retained_gram_v1b.log 2>&1
for L in 16 24 32; do
  nice -n 10 $PY n1_bracket_v1b.py --L $L --m 0.2 --n_c 1 --n_K 2 --threads 8 --out $OUT > $OUT/n1_bracket_v1b_L${L}_m0.2.log 2>&1
done
nice -n 10 $PY n1_bracket_v1b.py --L 24 --m 0.0 --n_c 1 --n_K 2 --threads 8 --out $OUT > $OUT/n1_bracket_v1b_L24_m0.log 2>&1
echo done > $OUT/DONE_n12_v1b
