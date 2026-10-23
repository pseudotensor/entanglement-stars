#!/bin/bash
# engine_star_nulltest_v1 (engine spec 28.4 (d)): the ell null test E_16 - 5 E_8 + 4 E_4 on every ell triple of the six families' records,
# the lead's quoted values (STAR_family_statement_v3.md, FILE 2) checked. Box CPU (seconds). Builder analyst.
# Usage: bash code/launch_star_nulltest_v1.sh <output md under SfamSum/> (a new name; an existing file is never overwritten)
cd ${QI2_ROOT}/engine || exit 1
OUT=${1:-SfamSum/STAR_nulltest_v1.md}
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT"; exit 2; fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nice -n 10 ${QI2_ROOT}/python code/engine_star_nulltest_v1.py --check-statement SfamSum/STAR_family_statement_v3.md --md "$OUT" --families \
  "Sfam6 (0.2, 1/4)=Sfam6/star_m0.2_g0.25_ell8.json,Sfam6/star_m0.2_g0.25_ell8_ext.json,Sfam6/star_m0.2_g0.25_ell8_ext2.json" \
  "Sfam5 (0.1, 1/4)=Sfam5/star_m0.1_g0.25_ell8.json" \
  "Sfam7 (0.05, 1/4)=Sfam7/star_m0.05_g0.25_ell8.json" \
  "Sfam6 (0.2, 1/16)=Sfam6/star_m0.2_g0.0625_ell8.json,Sfam6/star_m0.2_g0.0625_ell8_ext.json" \
  "Sfam6 (0.1, 1/16)=Sfam6/star_m0.1_g0.0625_ell8.json,Sfam6/star_m0.1_g0.0625_ell8_ext.json" \
  "Sfam8 (0.1, 1/64)=Sfam8/star_star8box_m0.1_g0.015625_max.json"
