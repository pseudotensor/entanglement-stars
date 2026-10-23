#!/bin/bash
# run_n12_v1.sh: foundations N1/N2 checks on the box (CPU, nice 10, 8 threads). Outputs to ${QI2_ROOT}/.
set -e
PY=${QI2_ROOT}/python
OUT=${QI2_ROOT}/n12
mkdir -p $OUT
cd "$(dirname "$0")"
nice -n 10 $PY n2_retained_gram_v1.py --out $OUT > $OUT/n2_retained_gram_v1.log 2>&1
for L in 8 16 24 32; do
  nice -n 10 $PY n1_bracket_v1.py --L $L --m 0.2 --n_c 1 --n_K 2 --threads 8 --out $OUT > $OUT/n1_bracket_v1_L${L}_m0.2.log 2>&1
done
nice -n 10 $PY n1_bracket_v1.py --L 16 --m 0.0 --n_c 1 --n_K 2 --threads 8 --out $OUT > $OUT/n1_bracket_v1_L16_m0.log 2>&1
echo done > $OUT/DONE_n12_v1
