#!/bin/bash
# COLLAPSE_certificate_v1.2 (engine spec 32.4 (1); compose_certificate_v3.py): waits for the production's engine_certificate_v3 exit 0
# line (CERT/status_v3.txt) and the two ladder members' (COLL/status_ladder_chain.txt), then composes in the ruled order: the production,
# the sigma/a 8 member, the sigma/a 4 member, the r_s 32 set's four (tau 16, 8, 4 and reading B, as version 1.1). Run on the box from
# engine/ in the background; a status line per step in CERT/status_v3.txt. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
S=CERT/status_v3.txt
M8=c27_rs64_A_R2_tau8_ell8_prep4_avg; M4=c27_rs64_A_R2_tau8_ell4_prep4_avg
echo "CERT12 compose waiting for the production's and the two members' v3 tables $(date -u +%H:%M:%S) UTC" >> $S
while true; do
  p=$(grep -c "^CERT3 c19_rs64 exit 0 " $S)
  a=$(grep -c "^chain $M8 certificate_v3 exit 0 " COLL/status_ladder_chain.txt)
  b=$(grep -c "^chain $M4 certificate_v3 exit 0 " COLL/status_ladder_chain.txt)
  bad=$(grep -E "^CERT3 c19_rs64 exit [1-9]|^chain ($M8|$M4) certificate_v3 exit [1-9]" $S COLL/status_ladder_chain.txt | head -1)
  if [ -n "$bad" ]; then echo "CERT12 compose not started: $bad $(date -u +%H:%M:%S) UTC" >> $S; exit 3; fi
  [ "$p" -ge 1 ] && [ "$a" -ge 1 ] && [ "$b" -ge 1 ] && break
  sleep 60
done
nice -n 10 $PY code/compose_certificate_v3.py --spec CERT/SPEC_final4b_engine_turn40.md --validate CERT/certificate_v3_validate.md --production CERT/certificate_v3_c19_rs64.md --members CERT/certificate_v3_${M8}.md,CERT/certificate_v3_${M4}.md --rs32 CERT/certificate_v3_set17_rs32_A_tau16.md,CERT/certificate_v3_set17_rs32_A_tau8.md,CERT/certificate_v3_set17_rs32_A_tau4.md,CERT/certificate_v3_set17_rs32_B_tau8.md --out CERT/COLLAPSE_certificate_v1.2.md > CERT/log_compose_v1_2.log 2>&1
echo "CERT12 compose exit $? $(date -u +%H:%M:%S) UTC" >> $S
