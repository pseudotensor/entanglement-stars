#!/bin/bash
# STAR_family_handoff_v2 (engine spec 26.4, 26.8): engine_star_handoff_v3 on the five families with the lead's statement at the head. Box CPU. Builder analyst.
cd ${QI2_ROOT}/engine
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nice -n 10 ${QI2_ROOT}/python code/engine_star_handoff_v3.py --statement SfamSum/STAR_family_statement_v2.md --md SfamSum/STAR_family_handoff_v2.md --families \
  "Sfam6 (0.2, 1/4)=Sfam6/star_m0.2_g0.25_ell8.json,Sfam6/star_m0.2_g0.25_ell8_ext.json,Sfam6/star_m0.2_g0.25_ell8_ext2.json" \
  "Sfam5 (0.1, 1/4)=Sfam5/star_m0.1_g0.25_ell8.json" \
  "Sfam7 (0.05, 1/4)=Sfam7/star_m0.05_g0.25_ell8.json" \
  "Sfam6 (0.2, 1/16)=Sfam6/star_m0.2_g0.0625_ell8.json,Sfam6/star_m0.2_g0.0625_ell8_ext.json" \
  "Sfam6 (0.1, 1/16)=Sfam6/star_m0.1_g0.0625_ell8.json" > SfamSum/log_star_handoff_v3.log 2>&1
echo "STAR handoff v3 exit $? $(date -u +%H:%M:%S) UTC" >> SUM5/status.txt
