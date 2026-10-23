#!/bin/bash
# STAR_family_handoff_v3 with statement 3.1 and the dense-table comparator (engine spec 29.6): engine_star_handoff_v5 (launcher version 1 of v5; was v4) on the six families (the (0.1, 1/64) member the sixth), the head
# composed from STAR_family_statement_v3.md and v2.1's first four paragraphs, the fields definition per record, the null-test column and the
# native-stress turning points of engine_ovpg_v3 (OVPG3/ovpg3_v1.json). Box CPU, nice 10, 8 threads. Builder analyst.
# Usage: bash code/launch_star_handoff_v4_v1.sh <output md under SfamSum/> [final]
#   test print:  bash code/launch_star_handoff_v4_v1.sh SfamSum/STAR_family_handoff_v3_test.md
#   final print: bash code/launch_star_handoff_v4_v1.sh SfamSum/STAR_family_handoff_v3.md final
#     ("final" requires the (0.1, 1/64) record's ell_control entry and also prints the null test to SfamSum/STAR_nulltest_v1_final.md)
# An existing output is never overwritten.
cd ${QI2_ROOT}/engine || exit 1
OUT=${1:?output md}
PY=${QI2_ROOT}/python
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT"; exit 2; fi
if [ "$2" = "final" ]; then
  NC=$($PY -c "import json; print(len(json.load(open('Sfam8/star_star8box_m0.1_g0.015625_max.json')).get('ell_control') or []))" 2>/dev/null)
  case "$NC" in
    ''|0) echo "the (0.1, 1/64) ell control has not landed (ell_control entries: ${NC:-unreadable})"; exit 3;;
  esac
fi
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "STAR handoff v5 -> $OUT start $(date -u +%H:%M:%S) UTC" >> SfamSum/status_handoff_v4.txt
nice -n 10 $PY code/engine_star_handoff_v5.py --statement-v21 SfamSum/STAR_family_statement_v2.1.md --statement-v3 SfamSum/STAR_family_statement_v3.md --statement-v31 SfamSum/STAR_family_statement_v3.1.md \
  --ovpg3 OVPG3/ovpg3_v1.json --md "$OUT" --families \
  "Sfam6 (0.2, 1/4)=Sfam6/star_m0.2_g0.25_ell8.json,Sfam6/star_m0.2_g0.25_ell8_ext.json,Sfam6/star_m0.2_g0.25_ell8_ext2.json" \
  "Sfam5 (0.1, 1/4)=Sfam5/star_m0.1_g0.25_ell8.json" \
  "Sfam7 (0.05, 1/4)=Sfam7/star_m0.05_g0.25_ell8.json" \
  "Sfam6 (0.2, 1/16)=Sfam6/star_m0.2_g0.0625_ell8.json,Sfam6/star_m0.2_g0.0625_ell8_ext.json" \
  "Sfam6 (0.1, 1/16)=Sfam6/star_m0.1_g0.0625_ell8.json,Sfam6/star_m0.1_g0.0625_ell8_ext.json" \
  "Sfam8 (0.1, 1/64)=Sfam8/star_star8box_m0.1_g0.015625_max.json" > "${OUT%.md}.log" 2>&1
RC=$?
echo "STAR handoff v5 -> $OUT exit $RC $(date -u +%H:%M:%S) UTC" >> SfamSum/status_handoff_v4.txt
if [ "$2" = "final" ] && [ ! -e SfamSum/STAR_nulltest_v1_final.md ]; then
  bash code/launch_star_nulltest_v1.sh SfamSum/STAR_nulltest_v1_final.md > SfamSum/log_star_nulltest_v1_final.log 2>&1
  echo "null test final exit $? $(date -u +%H:%M:%S) UTC" >> SfamSum/status_handoff_v4.txt
fi
