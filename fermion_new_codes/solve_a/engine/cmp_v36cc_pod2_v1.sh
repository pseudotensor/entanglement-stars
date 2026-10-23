#!/bin/bash
# cmp_v36cc_pod2_v1.sh (solve builder 9, analyst; engine spec 51.1 (2)): the on gate's reading, light CPU: every 5 min while the v36c
# relaunch runs (and once after it ends), compare_member_floor_v2 --criterion twice of its records against the stopped v32 member's stored
# ones (records to 18.125; the gate's span 16.125 to 17.5), md and json into ${QI2_ROOT}/.
set -u
D=${QI2_ROOT}/v32cont_v36c; C=${QI2_ROOT}/code_v36c; M=${QI2_ROOT}/c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b
mkdir -p $D/cmp
one() { T=$(cat $D/CURRENT_TAG 2>/dev/null) || return; [ -e $D/collapse_$T.json ] || return
  cd $C && python compare_member_floor_v2.py --run $D/$T --ref $M --label-run v36c_resolve --label-ref v32_member --criterion twice \
    --md $D/cmp/cmp_$T.md --out-json $D/cmp/cmp_$T.json > $D/cmp/cmp_$T.stdout 2>&1; }
while ! grep -q "v36cc exit" $D/status.txt 2>/dev/null; do one; sleep 300; done
one; echo "final $(date -u '+%F %T UTC')" >> $D/cmp/done.txt
